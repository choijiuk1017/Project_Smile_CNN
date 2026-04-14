from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import torch
import torch.nn as nn
from torchvision import transforms

# =========================
# 설정 (학습 코드와 동일해야 함)
# =========================
MODEL_SAVE_PATH = "cnn_weights.pth"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


# =========================
# SimpleCNN (학습 코드와 동일!)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 클래스 이름 로드
# =========================
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

num_classes = len(class_names)


# =========================
# 모델 로드
# =========================
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()

print(f"Using device: {device}")


# =========================
# 전처리 (학습 코드와 동일)
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# =========================
# API: /predict
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. 이미지 바이트 받기
        image_bytes = request.data

        # 2. PIL 이미지로 변환
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 3. 전처리
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 4. 추론
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        # 5. 결과 추출
        best_idx = torch.argmax(probs).item()
        best_class = class_names[best_idx]
        best_conf = float(probs[best_idx].item())

        # 🔥 디버깅 출력
        print("\n===== Prediction =====")
        print(f"Class: {best_class}")
        print(f"Confidence: {best_conf:.4f}")

        # Top 결과도 출력
        sorted_indices = torch.argsort(probs, descending=True)
        for idx in sorted_indices:
            idx = idx.item()
            print(f"{class_names[idx]}: {probs[idx].item():.4f}")

        # 6. JSON 반환
        return jsonify({
            "class": best_class,
            "confidence": best_conf
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# =========================
# 서버 실행
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)