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
DATA_PATH = "lora_tutorial_hint_reasoning_policy.jsonl"
OUTPUT_DIR = "gemma_hint_lora"
MAX_LENGTH = 512

SYSTEM_PROMPT = """너는 공포 퍼즐 게임의 주인공 독백을 생성한다.

반드시 지켜라:
- 명령문으로 말하지 않는다.
- '~하자', '~해야 한다', '~봐야 한다'를 쓰지 않는다.
- 정답, 해결 방법, 행동 지시를 말하지 않는다.
- RAG에 없는 물체나 방법을 만들지 않는다.
- 비밀번호, 열쇠, 카드 같은 단어는 RAG나 기준 독백에 있을 때만 쓴다.
- 기준 독백의 의미를 유지한다.
- 짧고 불안한 독백처럼 말한다.
- 한 문장만 출력한다.

좋은 예:
'저건 그냥 지나칠 물건은 아닌 것 같다.'
'이 방에는 뭔가 남아 있을지도 모른다.'
'저 문은 그냥 열릴 것 같지 않다.'

나쁜 예:
'책상을 둘러보자.'
'문을 열려면 비밀번호가 필요하다.'
'단서를 찾아야 한다.'"""

DEFAULT_REASONING_POLICY = """기준 독백의 의미를 가장 우선한다.
scene은 현재 이미지 상황을 참고하기 위한 정보다.
rag는 기준 독백이 어떤 퍼즐 맥락인지 확인하기 위한 정보다.
기준 독백에 없는 새로운 행동, 위치, 물체, 해결 방법을 추가하지 않는다.
짧고 진중한 주인공 독백형으로 출력한다."""


def build_prompt(scene, rag, reference_answer, reasoning_policy=None):
    if reasoning_policy is None or str(reasoning_policy).strip() == "":
        reasoning_policy = DEFAULT_REASONING_POLICY

    return f"""<start_of_turn>user
{SYSTEM_PROMPT}

[장면]
{scene}

[RAG 정보]
{rag}

[기준 독백]
{reference_answer}

[추론 규칙]
{reasoning_policy}

기준 독백과 같은 의미로 자연스럽게 한 문장으로 바꿔라.<end_of_turn>
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

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


def build_language_model_targets():
    targets = []

    for i in range(35):
        base = f"language_model.layers.{i}"

        targets.append(f"{base}.self_attn.q_proj")
        targets.append(f"{base}.self_attn.k_proj")
        targets.append(f"{base}.self_attn.v_proj")
        targets.append(f"{base}.self_attn.o_proj")

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

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=build_language_model_targets(),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    model.train()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    def tokenize(example):
        scene = example["scene"]
        rag = example["rag"]
        reference_answer = example["reference_answer"]
        reasoning_policy = example.get("reasoning_policy", DEFAULT_REASONING_POLICY)

        prompt = build_prompt(
            scene=scene,
            rag=rag,
            reference_answer=reference_answer,
            reasoning_policy=reasoning_policy,
        )

        answer = example["answer"].strip() + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        answer_ids = tokenizer(
            answer,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)

        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train",
    )

    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
    )

    collator = CausalLMCollator(tokenizer)

    test_batch = collator([dataset[0]])
    test_batch = {
        k: v.to(model.device)
        for k, v in test_batch.items()
    }

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
        raise RuntimeError("LoRA gradient가 0입니다.")

    model.zero_grad()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
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