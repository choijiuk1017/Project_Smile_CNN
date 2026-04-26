import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# =========================
# 설정
# =========================
DATASET_PATH = "dataset"              # 학습 이미지 폴더 경로
MODEL_SAVE_PATH = "cnn_weights.pth"   # 학습 완료 후 모델 저장 파일
CLASS_NAMES_PATH = "class_names.json" # 클래스 이름 저장 파일

IMAGE_SIZE = 224      # 입력 이미지 크기
BATCH_SIZE = 16       # 한 번에 학습할 이미지 개수
EPOCHS = 35           # 전체 학습 반복 횟수
LEARNING_RATE = 0.001 # 학습률
TRAIN_RATIO = 0.8     # 학습용 데이터 비율
SEED = 42             # 랜덤 시드 고정

NUM_WORKERS = 2       # 데이터 로더 병렬 처리 수


# =========================
# SimpleCNN 모델 정의
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # 특징 추출부
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),  
            nn.ReLU(),
            nn.MaxPool2d(2),                              

            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             

            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             

            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                 
        )

        # 분류부
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)   # 특징 추출
        x = self.classifier(x) # 클래스 분류
        return x


# =========================
# 검증 함수
# =========================
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc


# =========================
# 메인 학습 함수
# =========================
def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version (torch): {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True

    # =========================
    # 데이터 전처리 / 증강
    # =========================
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # =========================
    # 데이터셋 로드
    # =========================
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"'{DATASET_PATH}' 폴더가 없습니다.")

    base_dataset = datasets.ImageFolder(root=DATASET_PATH)
    class_names = base_dataset.classes
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError("최소 2개 이상의 클래스 폴더가 필요합니다.")

    print("Classes:", class_names)
    print("Total images:", len(base_dataset))

    # 클래스 이름 저장
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # =========================
    # train / val 분리
    # =========================
    train_size = int(len(base_dataset) * TRAIN_RATIO)
    val_size = len(base_dataset) - train_size

    train_dataset, val_dataset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_dataset.dataset = datasets.ImageFolder(root=DATASET_PATH, transform=train_transform)
    val_dataset.dataset = datasets.ImageFolder(root=DATASET_PATH, transform=val_transform)

    use_pin_memory = torch.cuda.is_available()
    use_persistent_workers = NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers
    )

    # =========================
    # 모델 / 손실 함수 / 옵티마이저
    # =========================
    model = SimpleCNN(num_classes=num_classes).to(device)
    print("Model device:", next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =========================
    # 학습
    # =========================
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # 최고 검증 정확도 갱신 시 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nBest Val Acc: {best_val_acc:.2f}%")
    print(f"Model saved: {MODEL_SAVE_PATH}")
    print(f"Class names saved: {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()