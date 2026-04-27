import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH = "qwen3_hint_lora_50.jsonl"
OUTPUT_DIR = "./qwen3_hint_lora"

def format_example(example):
    user_text = f"""
{example["instruction"]}

[LLaVA 장면 분석]
{example["input"]["scene"]}

[현재 구역]
{example["input"]["area_id"]}

[RAG 문서]
{chr(10).join("- " + x for x in example["input"]["rag_docs"])}

규칙:
- 반드시 한국어로만 답한다.
- 반드시 1문장만 출력한다.
- RAG 문서에 없는 정보는 만들지 않는다.
- 장면에 보이지 않는 물체는 말하지 않는다.
- 정답을 직접 말하지 않는다.
- 출력은 힌트 문장만 작성한다.
"""

    messages = [
        {"role": "system", "content": "너는 공포 퍼즐 게임의 조사 힌트 생성기다."},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": example["output"]}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

dataset = dataset.map(
    lambda x: {"text": format_example(x)}
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=8,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    max_length=1024,
    packing=False,
    dataset_text_field="text",
    report_to="none",

    fp16=False,
    bf16=False,
    max_grad_norm=0.0
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA 학습 완료:", OUTPUT_DIR)