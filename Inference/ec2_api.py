from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import tempfile
import os
import time

from cnn_lstm import CNN_LSTM

# ===== SETTINGS =====
MODEL_PATH = "../outputs/training/best_model_2_1774971874.pth"
SEQUENCE_LENGTH = 16
IMG_SIZE = 224
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD MODEL =====
model = CNN_LSTM(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded!")

# ===== MEDIAPIPE =====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== CLASS LABELS =====
CLASSES = sorted([
    "5 dollars", "advertise", "after", "alarm", "all of_sudden",
    "alphabet", "and", "anyone", "apple", "baseball cap",
    "basement", "bath", "blood", "borrow", "bottle", "braids",
    "brother", "call tty", "capture", "category", "catholic",
    "cemetery", "chain", "channel", "cheek", "chocolate",
    "cigarette", "clear", "close", "cocacola", "cocaine",
    "complex", "cookie", "cover up", "depend on", "disappear",
    "dress", "each", "egg beater", "egypt", "embarrass",
    "explanation", "fail", "falling asleep", "farm", "fault",
    "favorite", "few", "filter", "finish", "four", "friendly",
    "future", "greece", "green", "hammer", "hello", "hope",
    "husband", "important", "king", "later", "laugh", "lighter",
    "lipstick", "look appearance", "maybe", "morning", "motorcycle",
    "mustache", "north", "other", "pepsi", "person", "play",
    "rain", "sew", "sink", "skate", "society", "speakers",
    "sshh", "steal", "stomach", "strange", "surprise", "suspect",
    "temptation", "text", "that", "things", "travel",
    "very interested", "vote", "wear", "which", "why", "wish",
    "work", "worm"
])

app = Flask(__name__)

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    frame_indices = set(
        np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)
    )

    frames = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in frame_indices:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            DrawingSpec = mp.solutions.drawing_utils.DrawingSpec
            left = DrawingSpec(color=(255, 0, 0), thickness=2)
            right = DrawingSpec(color=(0, 0, 255), thickness=2)
            pose = DrawingSpec(color=(0, 255, 255), thickness=2)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS, pose, pose)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS, left, left)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS, right, right)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(transform(img))

        frame_id += 1

    cap.release()

    # pad if needed
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])

    frames = frames[:SEQUENCE_LENGTH]
    return torch.stack(frames).unsqueeze(0).to(DEVICE)

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    file = request.files["video"]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    start_time = time.time()

    tensor = preprocess_video(tmp_path)
    os.unlink(tmp_path)

    if tensor is None:
        return jsonify({"error": "Could not process video"}), 500

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASSES[predicted.item()]

    latency = time.time() - start_time

    return jsonify({
        "prediction": predicted_class,
        "latency_sec": latency
    })

@app.route("/", methods=["GET"])
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ASL Recognition - EC2</title>
        <style>
            body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
            h1 { color: #232f3e; }
            .result { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 8px; }
            button { background: #ff9900; color: white; padding: 10px 20px; border: none; cursor: pointer; border-radius: 4px; }
            .label { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>ASL Sign Language Recognition</h1>
        <p>Platform: <strong>EC2 (g5.xlarge)</strong></p>
        <form id="uploadForm">
            <input type="file" id="videoFile" accept="video/*" required><br><br>
            <button type="submit">Translate Sign</button>
        </form>
        <div id="result" class="result" style="display:none;">
            <p><span class="label">Prediction:</span> <span id="prediction"></span></p>
            <p><span class="label">Latency:</span> <span id="latency"></span> seconds</p>
        </div>
        <script>
            document.getElementById("uploadForm").onsubmit = async function(e) {
                e.preventDefault();
                document.getElementById("result").style.display = "none";
                const formData = new FormData();
                formData.append("video", document.getElementById("videoFile").files[0]);
                const response = await fetch("/predict", { method: "POST", body: formData });
                const data = await response.json();
                document.getElementById("prediction").innerText = data.prediction;
                document.getElementById("latency").innerText = data.latency_sec.toFixed(2);
                document.getElementById("result").style.display = "block";
            };
        </script>
    </body>
    </html>
    '''
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)