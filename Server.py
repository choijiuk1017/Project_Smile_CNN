from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import torch
import logging
import re

from transformers import (
    LlavaOnevisionProcessor,
    LlavaOnevisionForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

logging.getLogger("transformers").setLevel(logging.ERROR)

KNOWLEDGE_BASE_PATH = "puzzle_docs.json"

LLAVA_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
LLM_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LORA_ADAPTER_PATH = "./qwen3_hint_lora"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = Flask(__name__)

with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

print("Loading LLaVA...")
llava_processor = LlavaOnevisionProcessor.from_pretrained(LLAVA_MODEL_ID)

llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    LLAVA_MODEL_ID,
    dtype=dtype,
    device_map="auto"
)
llava_model.eval()

print("Loading Qwen3 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    LORA_ADAPTER_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Qwen3 Base...")
base_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    dtype=dtype,
    device_map="auto",
    trust_remote_code=True
)

print("Loading Qwen3 LoRA Adapter...")
llm_model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_PATH
)
llm_model.eval()


def retrieve_documents_by_area(area_id, max_spoiler_level=1):
    return [
        doc for doc in knowledge_base
        if doc.get("area_id") == area_id
        and doc.get("spoiler_level", 999) <= max_spoiler_level
    ]


def build_rag_context(rag_docs):
    if not rag_docs:
        return "관련 문서 없음"

    lines = []

    for doc in rag_docs:
        for fact in doc.get("facts", []):
            lines.append(f"- FACT: {fact}")
        for rule in doc.get("hint_rules", []):
            lines.append(f"- RULE: {rule}")
        for forbidden in doc.get("forbidden_assumptions", []):
            lines.append(f"- FORBIDDEN: {forbidden}")

        content = doc.get("content")
        if content:
            lines.append(f"- CONTENT: {content}")

    return "\n".join(lines) if lines else "관련 문서 없음"


def resize_image(image, max_size=384):
    image.thumbnail((max_size, max_size))
    return image


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
            max_new_tokens=80,
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
        scene_text = "Visible scene elements are unclear, but only visible objects should be considered."

    return scene_text


def extract_json_object(text):
    clean = text.strip()

    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean).strip()
        clean = re.sub(r"```$", "", clean).strip()

    try:
        return json.loads(clean)
    except Exception:
        pass

    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def generate_hint_with_qwen3_lora(scene_text, area_id, rag_docs):
    rag_context = build_rag_context(rag_docs)

    messages = [
        {
            "role": "system",
            "content": (
                "너는 공포 퍼즐 게임의 조사 보조 AI다. "
                "LLaVA 장면 설명과 RAG 문서를 종합해서 플레이어에게 보여줄 힌트만 작성한다. "
                "절대 JSON, 목록, 제목, 마크다운을 출력하지 않는다."
            )
        },
        {
            "role": "user",
            "content": f"""
너는 공포 퍼즐 게임의 조사 보조 AI다.

너의 목표는 분위기 잡는 게 아니라,
장면에 보이는 요소를 기반으로 플레이어가 다음 행동을 떠올리게 만드는 것이다.

규칙:
- 반드시 한국어 1문장만 출력한다.
- 감성적인 표현, 시적인 표현 금지
- 모호한 표현 금지 ("뭔가", "어딘가", "무언가" 금지)
- 장면에 보이는 요소를 반드시 포함한다 (시체, 혈흔, 문, 테이블 등)
- scene에 없는 것은 절대 말하지 않는다
- RAG 문서는 참고해서 연결만 한다
- 정답을 직접 말하지 않는다
- 문장은 "관찰 → 의미 연결" 구조로 작성한다

좋은 예:
- 쓰러진 사람과 문이 함께 보인다면, 이 문이 그냥 지나가는 길은 아닌 것 같다.
- 테이블 위에 남은 흔적을 보면, 여기에서 확인해야 할 단서가 더 있는 것 같다.

나쁜 예:
- 이곳에 남은 흔적이 모든 것을 말해주고 있다
- 무언가 이상한 기운이 느껴진다

[LLaVA 장면 분석]
{scene_text}

[RAG 문서]
{rag_context}

위 정보를 기반으로 힌트 1문장을 작성하라.
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
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    hint = tokenizer.decode(generated, skip_special_tokens=True).strip()

    hint = hint.replace("\n", " ").strip()

    for prefix in ["힌트:", "답변:", "출력:", "-", "•"]:
        if hint.startswith(prefix):
            hint = hint[len(prefix):].strip()

    for sep in ["。", ".", "!", "?"]:
        if sep in hint:
            hint = hint.split(sep)[0].strip() + "."
            break

    banned_words = [
        "JSON", "목록", "마크다운", "출력", "예시",
        "날짜", "시간", "번호", "암호", "비밀번호", "순서", "패턴"
    ]

    if len(hint) < 5 or any(word in hint for word in banned_words):
        hint = "이 장면에 남은 흔적들을 보면, 주변을 더 살펴봐야 할 것 같다."

    if not hint.endswith("."):
        hint += "."

    return hint

def make_hint_from_reasoning(reasoning):
    summary = reasoning.get("reasoning_summary", "").strip()

    banned_words = [
        "날짜", "시간", "순서", "패턴", "표시", "배열",
        "암호", "비밀번호", "번호", "코드", "기호",
        "목록", "제목", "마크다운", "출력", "예시",
        "JSON", "필드"
    ]

    if len(summary) < 5:
        return "이 장면에서 보이는 단서들을 기준으로 주변을 더 살펴봐야 할 것 같다."

    if any(word in summary for word in banned_words):
        return "이 장면에서 보이는 단서들을 기준으로 주변을 더 살펴봐야 할 것 같다."

    summary = summary.replace("\n", " ").strip()

    for sep in ["。", ".", "!", "?"]:
        if sep in summary:
            summary = summary.split(sep)[0].strip()

    if summary.endswith(("다", "요", "같다", "것 같다")):
        return summary + "."

    return summary + "."


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


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
        hint = generate_hint_with_qwen3_lora(scene, area_id, rag_docs)

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


if __name__ == "__main__":
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Server Start")
    app.run(host="0.0.0.0", port=5000)