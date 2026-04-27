from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import torch
import logging

from transformers import (
    LlavaOnevisionProcessor,
    LlavaOnevisionForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel

logging.getLogger("transformers").setLevel(logging.ERROR)

# =========================
# 설정
# =========================
KNOWLEDGE_BASE_PATH = "puzzle_docs.json"

LLAVA_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
LORA_PATH = "gemma_hint_lora"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = Flask(__name__)

# =========================
# RAG 문서 로드
# =========================
with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# =========================
# LLaVA 로드
# =========================
print("Loading LLaVA...")

llava_processor = LlavaOnevisionProcessor.from_pretrained(LLAVA_MODEL_ID)

llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    LLAVA_MODEL_ID,
    dtype=dtype,
    device_map="auto",
)

llava_model.eval()

# =========================
# Gemma + LoRA 로드
# =========================
print("Loading Gemma Base...")

tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_MODEL_ID,
    dtype=dtype,
    device_map="auto",
)

print("Loading Gemma LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

model.eval()

# =========================
# RAG 검색
# =========================
def retrieve_documents_by_area(area_id, max_spoiler_level=1):
    return [
        doc for doc in knowledge_base
        if doc.get("area_id") == area_id
        and doc.get("spoiler_level", 999) <= max_spoiler_level
    ]

# =========================
# RAG 텍스트 구성
# 중요:
# hint_rules는 넣지 않음
# facts만 넣어야 모델이 규칙을 그대로 복사하지 않음
# =========================
def build_rag_context(rag_docs):
    if not rag_docs:
        return "관련 문서 없음"

    lines = []

    for doc in rag_docs:
        for fact in doc.get("facts", []):
            lines.append(f"- {fact}")

    return "\n".join(lines)

# =========================
# 이미지 크기 축소
# =========================
def resize_image(image, max_size=384):
    image.thumbnail((max_size, max_size))
    return image

# =========================
# LLaVA 분석
# =========================
def analyze_image_with_llava(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = resize_image(image)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Describe only visible objects in one short sentence. "
                        "No guessing, no story, no gameplay."
                    ),
                },
            ],
        }
    ]

    prompt = llava_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )

    inputs = llava_processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(llava_model.device)

    with torch.no_grad():
        output = llava_model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    scene_text = llava_processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return scene_text

# =========================
# Gemma LoRA 힌트 생성
# 학습 때 사용한 프롬프트 형식과 맞춤
# =========================
SYSTEM_PROMPT = """너는 공포 퍼즐 게임의 조사 보조 AI다.
한국어 한 문장만 출력한다.
RAG 문서를 그대로 복사하지 않는다.
장면에 실제로 보이는 요소만 사용한다.
문서 사실은 참고만 하고 자연스러운 힌트로 바꾼다.
주인공의 독백형으로 말하듯이 힌트를 자연스럽게 말한다.
현재 상황에 따라 항상 같은 답을 주지 않도록 주의한다.
정답을 직접 말하지 않는다.
'무언가', '어딘가'를 쓰지 않는다."""

def build_gemma_prompt(scene_text, rag_context):
    return f"""<start_of_turn>user
{SYSTEM_PROMPT}

[장면]
{scene_text}

[참고 문서]
{rag_context}

힌트 문장만 작성해라.<end_of_turn>
<start_of_turn>model
"""

def clean_hint(full_text, prompt):
    hint = full_text

    if prompt in hint:
        hint = hint.replace(prompt, "")

    if "<start_of_turn>model" in hint:
        hint = hint.split("<start_of_turn>model")[-1]

    if "<end_of_turn>" in hint:
        hint = hint.split("<end_of_turn>")[0]

    hint = hint.replace("\n", " ").strip()

    # 앞에 불필요한 기호 제거
    while hint.startswith("-") or hint.startswith(":"):
        hint = hint[1:].strip()

    # 한 문장만 남김
    for sep in [".", "!", "?"]:
        if sep in hint:
            hint = hint.split(sep)[0] + sep
            break

    if len(hint) < 5:
        hint = "현재 장면에서 보이는 단서를 기준으로 주변을 더 확인해야 합니다."

    return hint

def generate_hint(scene_text, rag_context):
    prompt = build_gemma_prompt(scene_text, rag_context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.35,
            top_p=0.85,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=False,
    )

    hint = clean_hint(result, prompt)

    return hint

# =========================
# API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("REQUEST RECEIVED")

        area_id = request.headers.get("X-Area-Id", "Unknown")

        image_bytes = request.data

        if not image_bytes:
            return jsonify({"error": "이미지 없음"}), 400

        rag_docs = retrieve_documents_by_area(area_id)
        rag_context = build_rag_context(rag_docs)

        scene = analyze_image_with_llava(image_bytes)
        hint = generate_hint(scene, rag_context)

        print("AREA:", area_id)
        print("SCENE:", scene)
        print("RAG:", rag_context)
        print("HINT:", hint)

        return jsonify({
            "scene": scene,
            "hint": hint,
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# =========================
# 실행
# =========================
if __name__ == "__main__":
    print(f"Using device: {device}")
    print("Server Start")
    app.run(host="0.0.0.0", port=5000)