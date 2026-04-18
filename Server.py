from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import torch
import torch.nn as nn
from torchvision import transforms, models

MODEL_SAVE_PATH = "cnn_weights.pth"
CLASS_NAMES_PATH = "class_names.json"
KNOWLEDGE_BASE_PATH = "puzzle_docs.json"
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


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
# 힌트 생성
# =========================
def generate_hint_from_doc(doc):
    if not doc:
        return "관련 단서를 찾지 못했다."
    return doc.get("content", "")


# =========================
# HTTP 엔드포인트
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 언리얼에서 area_id, spoiler_level을 헤더로 보낸다고 가정
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

        hint = generate_hint_from_doc(best_doc)

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