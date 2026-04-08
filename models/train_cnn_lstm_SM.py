import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import time
import json

from cnn_lstm import CNN_LSTM


# ===== SETTINGS =====

TRAIN_DIR = os.environ["SM_CHANNEL_TRAIN"]
VAL_DIR = os.environ["SM_CHANNEL_VAL"]
TEST_DIR = os.environ["SM_CHANNEL_TEST"]

IMG_SIZE = 224
SEQUENCE_LENGTH = 16
BATCH_SIZE = 4
EPOCHS = 70
INSTANCE_TYPE = "g5.xlarge"
INSTANCE_PRICE_PER_HOUR = 1.41     # SageMaker approx
NUM_RUNS = 10

os.makedirs("checkpoints", exist_ok=True)
OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===== DATASET =====

class VideoDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.samples = []
        self.transform = transform

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:

            class_path = os.path.join(root_dir, cls)

            for video in os.listdir(class_path):

                video_path = os.path.join(class_path, video)

                if os.path.isdir(video_path):
                    self.samples.append((video_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        video_path, label = self.samples[idx]

        frames = sorted(os.listdir(video_path))

        if len(frames) < SEQUENCE_LENGTH:
            frames = frames + [frames[-1]] * (SEQUENCE_LENGTH - len(frames))
        else:
            frames = frames[:SEQUENCE_LENGTH]

        images = []

        for f in frames:
            img = Image.open(os.path.join(video_path, f)).convert("RGB")

            if self.transform:
                img = self.transform(img)

            images.append(img)

        images = torch.stack(images)

        return images, label


# ===== TRANSFORMS =====

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ===== DATA LOADERS =====

train_dataset = VideoDataset(TRAIN_DIR, transform)
val_dataset = VideoDataset(VAL_DIR, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    pin_memory=True
)


for run in range(NUM_RUNS):

    print(f"\n===== RUN {run+1}/{NUM_RUNS} =====")

    run_id = f"{run}_{int(time.time())}"

    # ===== MODEL RESET =====
    model = CNN_LSTM(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


    best_val_acc = 0

    train_start_time = time.time()
    epoch_times = []
    epoch_logs = []

    # ===== TRAIN LOOP =====
    for epoch in range(EPOCHS):

        epoch_start = time.time()

        model.train()

        total_loss = 0
        correct_train = 0
        total_train = 0

        for videos, labels in train_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Time: {epoch_time:.2f}s")

        # ===== VALIDATION =====
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in val_loader:

                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total

        epoch_logs.append({
            "epoch": epoch + 1,
            "epoch_time_sec": epoch_time,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "loss": avg_loss
        })

        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model_{run_id}.pth")  
            print("Best model saved!")

    print("Training finished")

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    train_total_time = time.time() - train_start_time
    training_cost = (train_total_time / 3600) * INSTANCE_PRICE_PER_HOUR

    print(f"Average Epoch Time: {avg_epoch_time:.2f} sec")
    print(f"Total Training Time: {train_total_time:.2f} sec")
    print(f"Training Cost: ${training_cost:.4f}")

    # ===== TEST =====
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model_{run_id}.pth", map_location=device))
    test_dataset = VideoDataset(TEST_DIR, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in test_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    print(f"TEST Accuracy: {test_acc:.4f}")

    # ===== SAVE RESULTS =====
    results = {
        "run_id": run_id,
        "instance_type": INSTANCE_TYPE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_time_sec": train_total_time,
        "training_cost": training_cost,
        "avg_epoch_time": avg_epoch_time,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc
    }

    with open(f"{OUTPUT_DIR}/training_results_{run_id}.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(f"{OUTPUT_DIR}/epoch_logs_{run_id}.json", "w") as f:
        json.dump(epoch_logs, f, indent=4)

    # ===== CLEAN GPU =====
    if torch.cuda.is_available():
        torch.cuda.empty_cache()