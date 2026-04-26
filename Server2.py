from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import torch
import logging
import os

from transformers import (
    LlavaOnevisionProcessor,
    LlavaOnevisionForConditionalGeneration
)

from openai import OpenAI

logging.getLogger("transformers").setLevel(logging.ERROR)

# =========================
# 설정
# =========================
KNOWLEDGE_BASE_PATH = "puzzle_docs.json"

LLAVA_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
OPENAI_MODEL_ID = "gpt-4.1-mini"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = Flask(__name__)

# =========================
# OpenAI 클라이언트
# =========================
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

client = OpenAI(api_key=api_key)

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
                        "Describe only what is visibly present in the image. "
                        "Do not infer gameplay, objectives, solutions, story, danger, or player actions. "
                        "Do not say 'the player must', 'suggesting', 'trapped', or 'navigate'. "
                        "Do not guess that an object is a keycard unless it is clearly visible as a card. "
                        "Mention only visible objects, colors, positions, and scene elements. "
                        "Write one short English sentence."
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
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=llava_processor.tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    scene_text = llava_processor.decode(generated, skip_special_tokens=True).strip()

    banned_phrases = [
        "player must",
        "navigate",
        "trapped",
        "suggesting",
        "horror game",
        "dangerous situation",
        "solve the puzzle"
    ]

    lowered = scene_text.lower()
    if any(p in lowered for p in banned_phrases):
        scene_text = scene_text.replace("The scene is a horror game where ", "")
        scene_text = scene_text.replace("the player must navigate through ", "")
        scene_text = scene_text.replace("The player must navigate through ", "")
        scene_text = scene_text.replace("suggesting they are trapped in a dangerous situation", "")

    return scene_text

# =========================
# 2단계: OpenAI 힌트 생성
# =========================
def generate_hint_with_openai(scene_text, area_id, rag_docs):
    rag_context = "\n".join(
        [f"- {doc.get('content', '')}" for doc in rag_docs]
    ) or "관련 문서 없음"

    instructions = """
너는 공포 퍼즐 게임의 조사 힌트 생성기이다.

규칙:
1. 반드시 한국어로만 답한다.
2. 반드시 1문장만 출력한다.
3. RAG 문서에 없는 정보는 절대 만들지 않는다.
4. 날짜, 시간, 번호, 암호, 순서, 패턴 같은 정보는 문서에 없으면 말하지 않는다.
5. 정답을 직접 말하지 않는다.
6. 단순 장면 설명이 아니라 플레이어가 다음 조사 방향을 유추할 수 있는 힌트를 작성한다.
7. 목록, 제목, 대괄호, 마크다운을 쓰지 않는다.
8. 출력은 힌트 문장만 작성한다.
9. 플레이어의 독백형 말투로 작성한다.
10. RAG 문서에서 참고를 하되, 내용을 전부 말할 필요는 없고 보이는 내용만 말하라.
11. 공포에 질린 말투로 말하라, 단 말투가 진중해야한다, 감탄사나 이런 내용은 어울리지 않는다.
"""

    user_prompt = f"""
[LLaVA 장면 분석]
{scene_text}

[현재 구역]
{area_id}

[RAG 문서]
{rag_context}

위 정보만 사용해서 플레이어에게 보여줄 힌트 1문장을 작성하라.
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL_ID,
            instructions=instructions,
            input=user_prompt,
            max_output_tokens=80
        )

        hint = response.output_text.strip()

        hint = hint.replace("\n", " ").strip()

        if "힌트:" in hint:
            hint = hint.split("힌트:")[-1].strip()

        banned = [
            "날짜", "시간", "순서", "패턴", "표시", "배열",
            "암호", "비밀번호", "번호", "코드", "기호"
        ]

        if any(word in hint for word in banned):
            hint = "주변을 더 살펴봐야겠어."

        if len(hint) < 5:
            hint = "이 장면의 흔적과 주변 단서가 서로 연결되어 있을 가능성이 있다."

        return hint

    except Exception as e:
        print("OpenAI ERROR:", str(e))
        return "이 장면의 흔적과 주변 단서가 서로 연결되어 있을 가능성이 있다."

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
        hint = generate_hint_with_openai(scene, area_id, rag_docs)

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