from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from openai import OpenAI

MODEL_SAVE_PATH = "cnn_weights.pth"
CLASS_NAMES_PATH = "class_names.json"
KNOWLEDGE_BASE_PATH = "puzzle_docs.json"
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


# =========================
# OpenAI 클라이언트 준비
# =========================
# PowerShell 예시:
# $env:OPENAI_API_KEY="여기에_발급받은_API_키"
# python app.py
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

client = OpenAI(api_key=api_key)


# =========================
# 클래스 이름 로드
# =========================
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

num_classes = len(class_names)


# =========================
# 지식 문서 로드
# =========================
with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)


# =========================
# 모델 로드
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()


# =========================
# 전처리
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =========================
# 문서 1개만 검색
# =========================
def retrieve_best_document(predicted_class: str, area_id: str = None, max_spoiler_level: int = 1):
    scored_docs = []

    for doc in knowledge_base:
        score = 0

        doc_area = doc.get("area_id")
        doc_labels = doc.get("labels", [])
        doc_spoiler = doc.get("spoiler_level", 999)
        content = doc.get("content", "").lower()

        # 스포일러 레벨 제한
        if doc_spoiler > max_spoiler_level:
            continue

        # 가장 중요한 조건: 예측 클래스와 문서 label이 일치해야 함
        if predicted_class not in doc_labels:
            continue

        # area_id까지 같으면 우선순위 상승
        if area_id and doc_area == area_id:
            score += 100

        # 클래스 일치 기본 점수
        score += 10

        # content 안에도 클래스명이 있으면 약간 보너스
        if predicted_class.lower() in content:
            score += 2

        # spoiler 낮을수록 약간 우선
        score += max(0, 5 - doc_spoiler)

        scored_docs.append((score, doc))

    # 점수 높은 순, spoiler 낮은 순 정렬
    scored_docs.sort(key=lambda x: (-x[0], x[1].get("spoiler_level", 999)))

    if scored_docs:
        return scored_docs[0][1]

    return None


# =========================
# 기존 문서 그대로 반환용 fallback
# =========================
def generate_hint_from_doc(doc):
    if not doc:
        return "관련 단서를 찾지 못했다."
    return doc.get("content", "")


# =========================
# OpenAI를 이용한 힌트 생성
# =========================
def generate_hint_with_llm(predicted_class: str, confidence: float, area_id: str, doc: dict, max_spoiler_level: int):
    if not doc:
        return "이 장면만으로는 확실한 단서를 찾기 어렵다."

    doc_text = doc.get("content", "")
    puzzle_id = doc.get("puzzle_id", "")
    doc_labels = doc.get("labels", [])
    doc_spoiler = doc.get("spoiler_level", 1)

    system_prompt = """
당신은 공포 퍼즐 게임의 조사 힌트 생성기이다.

규칙:
1. 반드시 입력된 문서 내용만 바탕으로 힌트를 작성한다.
2. 문서에 없는 정보는 절대 추가하지 않는다.
3. 퍼즐 정답이나 해결법을 직접 말하지 않는다.
4. 허용된 스포일러 수준을 넘는 정보는 말하지 않는다.
5. confidence가 낮으면 단정하지 말고 조심스럽게 표현한다.
6. 한국어로만 작성한다.
7. 길이는 1~2문장으로 짧게 쓴다.
8. 게임 속 조사 문구나 주인공의 짧은 독백처럼 자연스럽게 쓴다.
"""

    user_prompt = f"""
[분류 결과]
predicted_class: {predicted_class}
confidence: {confidence:.4f}
area_id: {area_id}
allowed_spoiler_level: {max_spoiler_level}

[검색된 문서]
puzzle_id: {puzzle_id}
labels: {doc_labels}
doc_spoiler_level: {doc_spoiler}
doc_text: {doc_text}

위 정보만 사용해서 플레이어에게 보여줄 힌트 문장을 작성하라.
"""

    try:
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_output_tokens=100
        )

        hint_text = response.output_text.strip()

        if not hint_text:
            return doc_text

        return hint_text

    except Exception as e:
        print("OpenAI ERROR:", str(e))
        # LLM 실패 시 기존 문서 텍스트 그대로 반환
        return doc_text


# =========================
# HTTP 엔드포인트
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        area_id = request.headers.get("X-Area-Id", None)

        spoiler_level_header = request.headers.get("X-Spoiler-Level", "1")
        try:
            max_spoiler_level = int(spoiler_level_header)
        except ValueError:
            max_spoiler_level = 1

        image_bytes = request.data
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        best_idx = torch.argmax(probs).item()
        best_class = class_names[best_idx]
        best_conf = float(probs[best_idx].item())

        best_doc = retrieve_best_document(
            predicted_class=best_class,
            area_id=area_id,
            max_spoiler_level=max_spoiler_level
        )

        # OpenAI로 힌트 생성
        hint = generate_hint_with_llm(
            predicted_class=best_class,
            confidence=best_conf,
            area_id=area_id,
            doc=best_doc,
            max_spoiler_level=max_spoiler_level
        )

        print(f"Predicted Class: {best_class}")
        print(f"Confidence: {best_conf:.4f}")
        print(f"Area ID: {area_id}")
        print(f"Hint: {hint}")

        return jsonify({
            "class": best_class,
            "confidence": best_conf,
            "area_id": area_id,
            "hint": hint,
            "retrieved_doc": best_doc
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    app.run(host="0.0.0.0", port=5000)