from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import torch
import logging
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    LlavaOnevisionProcessor,
    LlavaOnevisionForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


logging.getLogger("transformers").setLevel(logging.ERROR)


# =========================
# PATH / MODEL CONFIG
# =========================

RAG_DATA_PATH = "lora_tutorial_hint_reasoning_policy.jsonl"

LLAVA_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
LORA_PATH = "gemma_hint_lora"

RETRIEVAL_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = Flask(__name__)


# =========================
# LOAD RAG DATA
# =========================

def load_rag_data(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            required_keys = [
                "scene_type",
                "scene",
                "rag",
                "reference_answer",
            ]

            for key in required_keys:
                if key not in item:
                    raise ValueError(
                        f"RAG data missing key '{key}' at line {line_number}"
                    )

            data.append(item)

    if not data:
        raise ValueError("RAG data is empty.")

    return data


rag_data = load_rag_data(RAG_DATA_PATH)


# =========================
# LOAD MODELS
# =========================

print("Loading LLaVA...")
llava_processor = LlavaOnevisionProcessor.from_pretrained(LLAVA_MODEL_ID)
llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    LLAVA_MODEL_ID,
    dtype=dtype,
    device_map="auto",
)
llava_model.eval()


print("Loading Retrieval Embedding Model...")
retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_ID)

rag_scene_texts = [item["scene"] for item in rag_data]

rag_scene_embeddings = retrieval_model.encode(
    rag_scene_texts,
    normalize_embeddings=True,
)


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
gemma_model = PeftModel.from_pretrained(base_model, LORA_PATH)
gemma_model.eval()


# =========================
# PROMPTS
# =========================

LLAVA_PROMPT = (
    "Describe only what is visibly present in the image. "
    "Do not infer gameplay, objectives, solutions, story, danger, or player actions. "
    "Do not say 'the player must', 'suggesting', 'trapped', or 'navigate'. "
    "Do not guess that an object is a keycard unless it is clearly visible as a card. "
    "Mention only visible objects, colors, positions, and scene elements. "
    "Write one short English sentence."
)


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

# =========================
# IMAGE / LLAVA
# =========================

def resize_image(image, max_size=384):
    image.thumbnail((max_size, max_size))
    return image


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
                    "text": LLAVA_PROMPT,
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
            max_new_tokens=80,
            do_sample=False,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]

    scene = llava_processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return scene


# =========================
# RAG RETRIEVAL
# =========================

def retrieve_best_rag(scene_text, top_k=3):
    query_embedding = retrieval_model.encode(
        [scene_text],
        normalize_embeddings=True,
    )

    scores = cosine_similarity(
        query_embedding,
        rag_scene_embeddings,
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        item = rag_data[idx]
        score = float(scores[idx])

        results.append(
            {
                "score": score,
                "scene_type": item["scene_type"],
                "scene": item["scene"],
                "rag": item["rag"],
                "reference_answer": item["reference_answer"],
                "reasoning_policy": item.get(
                    "reasoning_policy",
                    DEFAULT_REASONING_POLICY,
                ),
            }
        )

    return results


def select_primary_rag(rag_results, threshold=0.55):
    if not rag_results:
        return {
            "score": 0.0,
            "scene_type": "no_clue",
            "scene": "",
            "rag": "단서 없음",
            "reference_answer": "잘 모르겠군.",
            "reasoning_policy": DEFAULT_REASONING_POLICY,
        }

    primary = rag_results[0]

    if primary["score"] < threshold:
        return {
            "score": primary["score"],
            "scene_type": "no_clue",
            "scene": primary["scene"],
            "rag": "단서 없음",
            "reference_answer": "지금 보이는 것만으로는 확실한 단서를 찾기 어렵다.",
            "reasoning_policy": DEFAULT_REASONING_POLICY,
        }

    return primary


def build_rag_context(rag_results):
    lines = []

    for idx, item in enumerate(rag_results, start=1):
        lines.append(f"[검색 후보 {idx}]")
        lines.append(f"score: {item['score']:.4f}")
        lines.append(f"scene_type: {item['scene_type']}")
        lines.append(f"reference_scene: {item['scene']}")
        lines.append(f"rag: {item['rag']}")
        lines.append(f"reference_answer: {item['reference_answer']}")
        lines.append("")

    return "\n".join(lines).strip()


# =========================
# GEMMA GENERATION
# =========================

def build_gemma_prompt(scene, rag, reference_answer, reasoning_policy=None):
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

def clean_hint(text):
    hint = text.strip()

    remove_tokens = [
        "<end_thought>",
        "<end_sequence>",
        "<turn|>",
        "<eos>",
        "<end_of_turn>",
        "<start_of_turn>",
        "<end_start_turn_>",
        "<end_>",
        "<start_>",
        "model",
        "user",
    ]

    for token in remove_tokens:
        hint = hint.replace(token, "")

    for bad in [
        "힌트:",
        "독백:",
        "정답:",
        "출력:",
        "Answer:",
        "Hint:",
    ]:
        hint = hint.replace(bad, "")

    hint = hint.replace("\n", " ").strip()
    hint = hint.strip("\"'“”‘’<>_ ")

    while "  " in hint:
        hint = hint.replace("  ", " ")

    return hint


def generate_hint(scene_text, rag_results):
    primary_rag = select_primary_rag(rag_results)
    rag_context = build_rag_context(rag_results)

    prompt = build_gemma_prompt(
        scene=scene_text,
        rag=primary_rag["rag"],
        reference_answer=primary_rag["reference_answer"],
        reasoning_policy=primary_rag.get("reasoning_policy", DEFAULT_REASONING_POLICY),
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(gemma_model.device)

    with torch.no_grad():
        output = gemma_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.03,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][inputs["input_ids"].shape[-1]:]

    result = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    hint = clean_hint(result)

    return hint, primary_rag, rag_context


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

        scene = analyze_image_with_llava(image_bytes)

        rag_results = retrieve_best_rag(
            scene_text=scene,
            top_k=3,
        )

        hint, primary_rag, rag_context = generate_hint(
            scene_text=scene,
            rag_results=rag_results,
        )

        print("AREA:", area_id)
        print("SCENE:", scene)
        print("MATCHED_SCENE_TYPE:", primary_rag.get("scene_type", "unknown"))
        print("MATCHED_SCORE:", primary_rag.get("score", 0.0))
        print("MATCHED_REFERENCE_SCENE:", primary_rag.get("scene", ""))
        print("RAG:", primary_rag.get("rag", ""))
        print("REFERENCE_ANSWER:", primary_rag.get("reference_answer", ""))
        print("HINT:", hint)

        return jsonify(
            {
                "area": area_id,
                "scene": scene,
                "matched_scene_type": primary_rag.get("scene_type", "unknown"),
                "matched_score": primary_rag.get("score", 0.0),
                "matched_reference_scene": primary_rag.get("scene", ""),
                "rag": primary_rag.get("rag", ""),
                "reference_answer": primary_rag.get("reference_answer", ""),
                "hint": hint,
                "rag_candidates": rag_results,
            }
        )

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/test_rag", methods=["POST"])
def test_rag():
    try:
        data = request.get_json()

        if not data or "scene" not in data:
            return jsonify({"error": "scene 값이 없습니다."}), 400

        scene = data["scene"]

        rag_results = retrieve_best_rag(
            scene_text=scene,
            top_k=3,
        )

        primary_rag = select_primary_rag(rag_results)

        return jsonify(
            {
                "scene": scene,
                "matched_scene_type": primary_rag.get("scene_type", "unknown"),
                "matched_score": primary_rag.get("score", 0.0),
                "matched_reference_scene": primary_rag.get("scene", ""),
                "rag": primary_rag.get("rag", ""),
                "reference_answer": primary_rag.get("reference_answer", ""),
                "rag_candidates": rag_results,
            }
        )

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Loaded RAG docs: {len(rag_data)}")
    print("Server Start")
    app.run(host="0.0.0.0", port=5000)