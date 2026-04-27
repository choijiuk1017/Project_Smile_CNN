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

KNOWLEDGE_BASE_PATH = "puzzle_docs.json"
LLAVA_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
LORA_PATH = "gemma_hint_lora"

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
    device_map="auto",
)
llava_model.eval()

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
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

SYSTEM_PROMPT = """너는 공포 게임 속 주인공의 내면 독백을 생성한다.
입력으로는 현재 장면 설명, RAG 정답 정보, 추론 정책이 주어진다.
RAG는 정답지이지만 출력에 정답 물체, 위치, 해결 방법을 직접 말하지 않는다.
현재 장면에 보이는 요소와 RAG 정보를 연결해 주인공이 떠올릴 법한 의심, 판단, 불안을 한 문장으로 말한다.
설명문이나 안내문처럼 말하지 않는다.
말투는 짧고 진중한 독백형이다.
'무언가', '어딘가', '확인해봐야겠어'를 남발하지 않는다.
한국어 한 문장만 출력한다."""

DEFAULT_REASONING_POLICY = """현재 장면에 실제로 보이는 요소를 우선한다.
RAG 정답 정보는 참고하되 정답 물체, 정확한 위치, 해결 방법은 직접 말하지 않는다.
시체와 혈흔은 사건 흔적으로만 해석하고, 문을 여는 직접 정답과 강제로 연결하지 않는다.
문이나 장치가 보이면 상태 변화나 조건이 필요하다는 수준으로만 말한다.
휴게 공간이나 테이블 위 물체가 보이면 남겨진 물건의 의미를 의심하는 수준으로만 말한다.
진중한 주인공 독백으로 한 문장만 출력한다."""

def retrieve_documents_by_area(area_id, max_spoiler_level=1):
    return [
        doc for doc in knowledge_base
        if doc.get("area_id") == area_id and doc.get("spoiler_level", 999) <= max_spoiler_level
    ]

def build_rag_context(rag_docs):
    if not rag_docs:
        return "관련 문서 없음"
    lines = []
    for doc in rag_docs:
        for fact in doc.get("facts", []):
            lines.append(f"- {fact}")
    return "\n".join(lines)

def resize_image(image, max_size=384):
    image.thumbnail((max_size, max_size))
    return image

def analyze_image_with_llava(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = resize_image(image)

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe only visible objects in one short sentence. No guessing, no story, no gameplay."},
        ],
    }]

    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = llava_processor(images=image, text=prompt, return_tensors="pt").to(llava_model.device)

    with torch.no_grad():
        output = llava_model.generate(**inputs, max_new_tokens=40, do_sample=False)

    generated = output[0][inputs["input_ids"].shape[-1]:]
    return llava_processor.decode(generated, skip_special_tokens=True).strip()

def build_gemma_prompt(scene_text, rag_context, reasoning_policy):
    return f"""<start_of_turn>user
{SYSTEM_PROMPT}

[장면]
{scene_text}

[RAG 정답 정보]
{rag_context}

[추론 정책]
{reasoning_policy}

위 정보를 바탕으로 정답을 직접 말하지 않는 주인공의 독백형 힌트 한 문장만 작성해라.<end_of_turn>
<start_of_turn>model
"""

def clean_hint(full_text, prompt):
    hint = full_text.replace(prompt, "")
    if "<start_of_turn>model" in hint:
        hint = hint.split("<start_of_turn>model")[-1]
    if "<end_of_turn>" in hint:
        hint = hint.split("<end_of_turn>")[0]
    hint = hint.replace("\n", " ").strip()
    while hint.startswith("-") or hint.startswith(":"):
        hint = hint[1:].strip()

    for bad in ["힌트:", "독백:", "정답:", "출력:"]:
        hint = hint.replace(bad, "").strip()

    for sep in [".", "!", "?"]:
        if sep in hint:
            hint = hint.split(sep)[0] + sep
            break

    if len(hint) < 5:
        hint = "이 장면만으로는 부족하다… 더 봐야 한다."
    return hint

def generate_hint(scene_text, rag_context):
    prompt = build_gemma_prompt(scene_text, rag_context, DEFAULT_REASONING_POLICY)

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.45,
            top_p=0.85,
            repetition_penalty=1.18,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(output[0], skip_special_tokens=False)
    return clean_hint(result, prompt)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("REQUEST RECEIVED")
        area_id = request.headers.get("X-Area-Id", "Unknown")
        image_bytes = request.data
        if not image_bytes:
            return jsonify({"error": "이미지 없음"}), 400

        scene = analyze_image_with_llava(image_bytes)
        rag_docs = retrieve_documents_by_area(area_id)
        rag_context = build_rag_context(rag_docs)
        hint = generate_hint(scene, rag_context)

        print("AREA:", area_id)
        print("SCENE:", scene)
        print("HINT:", hint)

        return jsonify({"scene": scene, "hint": hint})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"Using device: {device}")
    print("Server Start")
    app.run(host="0.0.0.0", port=5000)
