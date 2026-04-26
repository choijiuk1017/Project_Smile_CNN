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
    AutoModelForCausalLM
)

from peft import PeftModel

logging.getLogger("transformers").setLevel(logging.ERROR)

# =========================
# 설정
# =========================
KNOWLEDGE_BASE_PATH = "puzzle_docs.json"

LLAVA_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
LLM_MODEL_ID = "google/gemma-2-2b-it"
LORA_ADAPTER_PATH = "./gemma_hint_lora"

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
    device_map="auto"
)
llava_model.eval()

# =========================
# Gemma 2 + LoRA 로드
# =========================
print("Loading Gemma 2 Base...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    dtype=dtype,
    device_map="auto"
)

print("Loading Gemma LoRA Adapter...")
llm_model = PeftModel.from_pretrained(
    llm_model,
    LORA_ADAPTER_PATH
)

llm_model.eval()

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
# 이미지 크기 축소
# =========================
def resize_image(image, max_size=384):
    image.thumbnail((max_size, max_size))
    return image

# =========================
# 1단계: LLaVA 이미지 분석
# =========================
def analyze_image_with_llava(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = resize_image(image, 384)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Describe this horror game scene in English. "
                        "Focus on visible puzzle-relevant clues only: blood, corpse, body, door, "
                        "keycard, card, lock, device, stairs, hallway, strange traces. "
                        "Do not solve the puzzle. "
                        "Write one short sentence."
                    )
                }
            ]
        }
    ]

    prompt = llava_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True
    )

    inputs = llava_processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(llava_model.device)

    with torch.no_grad():
        output = llava_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=llava_processor.tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    scene_text = llava_processor.decode(generated, skip_special_tokens=True)

    return scene_text.strip()

# =========================
# 2단계: Gemma LoRA 힌트 생성
# =========================
def generate_hint_with_gemma_lora(scene_text, area_id, rag_docs):
    rag_context = "\n".join(
        [f"- {doc.get('content', '')}" for doc in rag_docs]
    ) or "관련 문서 없음"

    messages = [
        {
            "role": "user",
            "content": f"""
너는 공포 퍼즐 게임의 힌트 생성기다.

[LLaVA 장면 분석]
{scene_text}

[현재 구역]
{area_id}

[RAG 문서]
{rag_context}

규칙:
- 반드시 한국어로만 답한다.
- 1문장만 출력한다.
- RAG 문서에 없는 정보는 만들지 않는다.
- 정답을 직접 말하지 않는다.
- 단순 설명이 아니라 다음 조사 방향을 암시한다.
- 목록, 제목, 대괄호, 마크다운을 쓰지 않는다.
- 플레이어가 독백하는 형태로 문장을 출력한다.

힌트 한 문장:
"""
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    with torch.no_grad():
        output = llm_model.generate(
            **inputs,
            max_new_tokens=45,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    hint = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # =========================
    # 후처리
    # =========================
    hint = hint.replace("\n", " ").strip()
    hint = hint.replace("[", "").replace("]", "").replace("-", "").strip()

    if "힌트:" in hint:
        hint = hint.split("힌트:")[-1].strip()

    if "힌트 한 문장:" in hint:
        hint = hint.split("힌트 한 문장:")[-1].strip()

    # 한 문장 제한
    for sep in ["。", ".", "!", "?"]:
        if sep in hint:
            hint = hint.split(sep)[0].strip() + sep
            break

    # 이상 출력 fallback
    banned = [
        "날짜", "시간", "순서", "패턴", "표시", "배열",
        "암호", "비밀번호", "번호", "코드", "기호",
        "목록", "제목", "마크다운", "규칙", "분석", "출력", "예시"
    ]

    if any(word in hint for word in banned):
        hint = "시체 주변의 흔적과 출입증처럼 보이는 물건이 막힌 문과 관련되어 있을 가능성이 있다."

    if len(hint) < 5:
        hint = "시체 주변의 흔적과 출입증처럼 보이는 물건이 막힌 문과 관련되어 있을 가능성이 있다."

    return hint

# =========================
# 테스트용
# =========================
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

# =========================
# API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("REQUEST RECEIVED")

        area_id = request.headers.get("X-Area-Id")
        if not area_id:
            return jsonify({"error": "AreaID 없음"}), 400

        try:
            spoiler_level = int(request.headers.get("X-Spoiler-Level", "1"))
        except ValueError:
            spoiler_level = 1

        image_bytes = request.data
        if not image_bytes:
            return jsonify({"error": "이미지 없음"}), 400

        rag_docs = retrieve_documents_by_area(area_id, spoiler_level)

        scene = analyze_image_with_llava(image_bytes)
        hint = generate_hint_with_gemma_lora(scene, area_id, rag_docs)

        print("SCENE:", scene)
        print("RAG DOCS:", rag_docs)
        print("HINT:", hint)

        return jsonify({
            "area_id": area_id,
            "scene": scene,
            "hint": hint
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# =========================
# 실행
# =========================
if __name__ == "__main__":
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Server Start")
    app.run(host="0.0.0.0", port=5000)