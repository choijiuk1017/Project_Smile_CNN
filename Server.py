from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import torch
import torch.nn as nn
from torchvision import transforms, models

MODEL_SAVE_PATH = "cnn_weights.pth"
CLASS_NAMES_PATH = "class_names.json"
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
# 학습 때 ResNet 정규화 썼으면 반드시 맞춰야 함
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =========================
# HTTP 엔드포인트
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_bytes = request.data
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        best_idx = torch.argmax(probs).item()
        best_class = class_names[best_idx]
        best_conf = float(probs[best_idx].item())

        print(f"Predicted Class: {best_class}")
        print(f"Confidence: {best_conf:.4f}")

        sorted_indices = torch.argsort(probs, descending=True)
        print("Top Results:")
        for idx in sorted_indices:
            idx = idx.item()
            print(f"{class_names[idx]}: {probs[idx].item():.4f}")

        return jsonify({
            "class": best_class,
            "confidence": best_conf
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    app.run(host="0.0.0.0", port=5000)