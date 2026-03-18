import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

from models.cnn_lstm import CNN_LSTM


# ===== SETTINGS =====

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

IMG_SIZE = 224
SEQUENCE_LENGTH = 24
BATCH_SIZE = 4
EPOCHS = 100

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        if not frames:
            raise RuntimeError(f"No frames found in: {video_path}")

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
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ===== DATA LOADERS =====

train_dataset = VideoDataset(TRAIN_DIR, transform)
val_dataset = VideoDataset(VAL_DIR, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True
)


# ===== MODEL =====

model = CNN_LSTM(num_classes=100).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=2,
    factor=0.5
)

best_val_acc = 0

# ===== TRAIN LOOP =====

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for videos, labels in train_loader:

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)

        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
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

    print(f"Validation Accuracy: {val_acc:.4f}")
    scheduler.step(val_acc)
    print("Current LR:", optimizer.param_groups[0]['lr'])
    
    # SAVE CHECKPOINT VARJE EPOCH
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, "checkpoints/latest.pt")
    
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, f"checkpoints/epoch_{epoch}.pt")



    # TA BORT gamla checkpoints lokalt
    keep_last = 5

    old_epoch = epoch - keep_last
    old_path = f"checkpoints/epoch_{old_epoch}.pt"

    if old_epoch >= 0 and os.path.exists(old_path):
        os.remove(old_path)

    # ===== SAVE BEST MODEL =====
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "outputs/best_cnn_lstm_model.pth"
        )

        print("Best model saved!")
        
print("Training finished")

# ===== TEST EVALUATION =====

# LOAD BEST MODEL (VIKTIGT)
model.load_state_dict(torch.load("outputs/best_cnn_lstm_model.pth"))

print("Running TEST evaluation...")



test_dataset = VideoDataset(TEST_DIR, transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True
)

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