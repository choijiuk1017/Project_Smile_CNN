import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

MODEL_ID = "google/gemma-2-2b-it"
DATASET_PATH = "hint_dataset.jsonl"
OUTPUT_DIR = "./gemma_hint_lora"

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_prompt(example):
    return f"""<start_of_turn>user
너는 공포 퍼즐 게임의 힌트 생성기다.

[LLaVA 장면 분석]
{example["scene"]}

[RAG 문서]
{example["rag"]}

규칙:
- 반드시 한국어로만 답한다.
- 1문장만 출력한다.
- RAG 문서에 없는 정보는 만들지 않는다.
- 정답을 직접 말하지 않는다.
- 단순 설명이 아니라 다음 조사 방향을 암시한다.

힌트 한 문장:
<end_of_turn>
<start_of_turn>model
{example["output"]}<end_of_turn>"""


def main():
    print("Using device:", device)

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        text = build_prompt(example)

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding="max_length"
        )

        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=20,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("LoRA saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()