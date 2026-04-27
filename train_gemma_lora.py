import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "google/gemma-4-E2B-it"
DATA_PATH = "lora_hint_dataset.jsonl"
OUTPUT_DIR = "gemma_hint_lora"
MAX_LENGTH = 1024

SYSTEM_PROMPT = """너는 공포 퍼즐 게임의 조사 보조 AI다.
한국어 한 문장만 출력한다.
RAG 문서를 그대로 복사하지 않는다.
장면에 실제로 보이는 요소만 사용한다.
문서 사실은 참고만 하고 자연스러운 힌트로 바꾼다.
정답을 직접 말하지 않는다.
'무언가', '어딘가'를 쓰지 않는다."""


def build_prompt(scene: str, rag: str) -> str:
    return f"""<start_of_turn>user
{SYSTEM_PROMPT}

[장면]
{scene}

[참고 문서]
{rag}

힌트 문장만 작성해라.<end_of_turn>
<start_of_turn>model
"""


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        token_type_ids_list = []
        mm_token_type_ids_list = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            input_ids_list.append(
                f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            attention_mask_list.append(
                f["attention_mask"] + [0] * pad_len
            )
            labels_list.append(
                f["labels"] + [-100] * pad_len
            )

            # Gemma 4 text-only 학습에서도 필요할 수 있음
            token_type_ids_list.append([0] * max_len)
            mm_token_type_ids_list.append([0] * max_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_list, dtype=torch.long),
            "mm_token_type_ids": torch.tensor(mm_token_type_ids_list, dtype=torch.long),
        }


def build_language_model_targets():
    targets = []

    # 로그 기준 Gemma 4 E2B-it는 language_model.layers.0~34 존재
    # 초반 0~14는 q/k/v/o 전부 존재
    # 후반 15~34는 일부 레이어에서 k_proj, v_proj가 로그에 안 보였음
    for i in range(35):
        base = f"language_model.layers.{i}"

        targets.append(f"{base}.self_attn.q_proj")
        targets.append(f"{base}.self_attn.o_proj")

        if i <= 14:
            targets.append(f"{base}.self_attn.k_proj")
            targets.append(f"{base}.self_attn.v_proj")

        targets.append(f"{base}.mlp.gate_proj")
        targets.append(f"{base}.mlp.up_proj")
        targets.append(f"{base}.mlp.down_proj")

    return targets


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto",
    )

    model.config.use_cache = False

    target_modules = build_language_model_targets()

    print("LoRA target count:", len(target_modules))
    print("LoRA target sample:")
    for t in target_modules[:20]:
        print(" ", t)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    model.train()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    def tokenize(example):
        prompt = build_prompt(example["scene"], example["rag"])
        answer = example["answer"].strip() + "<end_of_turn>"

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)

        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]

        label_count = sum(1 for x in labels if x != -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_count": label_count,
        }

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    print("Label count sample:")
    for i in range(min(5, len(dataset))):
        print(dataset[i]["label_count"])

    dataset = dataset.remove_columns(["label_count"])

    collator = CausalLMCollator(tokenizer)

    test_batch = collator([dataset[0]])
    test_batch = {
        k: v.to(model.device)
        for k, v in test_batch.items()
    }

    model.train()
    output = model(**test_batch)

    print("LOSS:", output.loss.item())
    print("LOSS REQUIRES GRAD:", output.loss.requires_grad)

    output.loss.backward()

    found_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_sum = 0.0 if param.grad is None else param.grad.abs().sum().item()

            if grad_sum > 0:
                print("GRAD CHECK OK:", name, grad_sum)
                found_grad = True
                break

    if not found_grad:
        raise RuntimeError("LoRA gradient가 0입니다. language_model에 LoRA가 연결되지 않았습니다.")

    model.zero_grad()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=8,
        logging_steps=1,
        save_strategy="epoch",
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()