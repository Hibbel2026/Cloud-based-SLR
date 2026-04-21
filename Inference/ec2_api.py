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
import json
import gzip
import boto3

from cnn_lstm import CNN_LSTM

# ===== SETTINGS =====
MODEL_PATH = "../outputs/training/best_model_2_1774971874.pth"
ENDPOINT_FILE = os.path.join(os.path.dirname(__file__), "endpoint_name.txt")
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

    tensor = preprocess_video(tmp_path)
    os.unlink(tmp_path)

    if tensor is None:
        return jsonify({"error": "Could not process video"}), 500

    start_time = time.time()

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASSES[predicted.item()]

    latency = time.time() - start_time

    return jsonify({
        "prediction": predicted_class,
        "latency_sec": latency,
        "platform": "EC2"
    })


@app.route("/predict/sagemaker", methods=["POST"])
def predict_sagemaker():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    if not os.path.exists(ENDPOINT_FILE):
        return jsonify({"error": "SageMaker endpoint not deployed (endpoint_name.txt not found)"}), 503

    with open(ENDPOINT_FILE, "r") as f:
        endpoint_name = f.read().strip()

    if not endpoint_name:
        return jsonify({"error": "endpoint_name.txt is empty"}), 503

    file = request.files["video"]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    tensor = preprocess_video(tmp_path)
    os.unlink(tmp_path)

    if tensor is None:
        return jsonify({"error": "Could not process video"}), 500

    # Serialize preprocessed tensor as gzip-compressed float32 bytes (~2-3MB vs 30MB JSON)
    tensor_bytes = tensor.cpu().numpy().astype("float32").tobytes()
    payload = gzip.compress(tensor_bytes)

    # Use named profile locally; fall back to instance role on EC2
    try:
        boto_session = boto3.session.Session(profile_name="HIBE", region_name="eu-north-1")
        boto_session.client("sts").get_caller_identity()  # verify credentials work
    except Exception:
        boto_session = boto3.session.Session(region_name="eu-north-1")

    sm_runtime = boto_session.client("sagemaker-runtime")

    # Measure only model inference latency
    infer_start = time.time()
    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/octet-stream",
            Body=payload
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    latency_ms = (time.time() - infer_start) * 1000

    result = json.loads(response["Body"].read().decode("utf-8"))

    return jsonify({
        "predicted_word": result["predicted_word"],
        "latency_ms": round(latency_ms, 2),
        "platform": "SageMaker"
    })

@app.route("/", methods=["GET"])
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ASL Sign Recognizer</title>
  <style>
    :root {
      --bg: #0d1117;
      --surface: #161b22;
      --surface2: #21262d;
      --border: #30363d;
      --text: #e6edf3;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-glow: rgba(88, 166, 255, 0.12);
      --success: #3fb950;
      --radius: 10px;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 48px 20px 64px;
    }

    header {
      text-align: center;
      margin-bottom: 36px;
    }

    header h1 {
      font-size: 1.85rem;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    header p {
      color: var(--text-muted);
      margin-top: 6px;
      font-size: 0.875rem;
      letter-spacing: 0.3px;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      width: 100%;
      max-width: 580px;
      margin-bottom: 14px;
    }

    .section-label {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--text-muted);
      margin-bottom: 14px;
    }

    /* Platform selector */
    .platform-btns {
      display: flex;
      gap: 10px;
    }

    .platform-btn {
      flex: 1;
      padding: 11px 16px;
      border-radius: 8px;
      border: 2px solid transparent;
      font-size: 0.875rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      cursor: pointer;
      transition: all 0.18s;
    }

    .platform-btn.active {
      background: rgba(255, 153, 0, 0.1);
      border-color: #f90;
      color: #f90;
    }

    .platform-btn:not(.active) {
      background: var(--surface2);
      border-color: var(--border);
      color: var(--text-muted);
    }

    .platform-btn:not(.active):hover {
      border-color: var(--accent);
      color: var(--text);
      opacity: 1;
    }

    .platform-btn small {
      font-weight: 400;
      font-size: 0.72rem;
      opacity: 0.85;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-size: 0.72rem;
      padding: 4px 10px;
      border-radius: 20px;
      background: rgba(63, 185, 80, 0.1);
      border: 1px solid rgba(63, 185, 80, 0.3);
      color: var(--success);
      margin-top: 14px;
    }

    .platform-label {
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-top: 8px;
    }

    .platform-label span {
      color: var(--accent);
      font-weight: 600;
    }

    .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 1.8s ease infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.35; }
    }

    /* Drop zone */
    .drop-zone {
      border: 2px dashed var(--border);
      border-radius: 8px;
      padding: 36px 20px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.18s, background 0.18s;
      position: relative;
    }

    .drop-zone:hover,
    .drop-zone.dragover {
      border-color: var(--accent);
      background: var(--accent-glow);
    }

    .drop-zone input[type="file"] {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
      width: 100%;
      height: 100%;
    }

    .drop-icon {
      font-size: 2.4rem;
      display: block;
      margin-bottom: 10px;
      pointer-events: none;
    }

    .drop-zone p {
      font-size: 0.9rem;
      color: var(--text-muted);
      pointer-events: none;
    }

    .drop-zone p .browse-link {
      color: var(--accent);
      font-weight: 500;
    }

    .drop-zone .formats {
      font-size: 0.72rem;
      margin-top: 6px;
      opacity: 0.55;
      pointer-events: none;
    }

    .file-selected {
      display: none;
      margin-top: 12px;
      font-size: 0.82rem;
      color: var(--success);
      word-break: break-all;
    }

    /* Preview */
    #preview-section {
      display: none;
      margin-top: 16px;
    }

    #video-preview {
      width: 100%;
      max-height: 280px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #000;
      display: block;
    }

    /* Buttons */
    .btn-row {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }

    .btn {
      padding: 10px 20px;
      border-radius: 8px;
      border: none;
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.18s;
      line-height: 1;
    }

    #preview-btn {
      display: none;
      background: var(--surface2);
      color: var(--text);
      border: 1px solid var(--border);
      white-space: nowrap;
    }

    #preview-btn:hover {
      border-color: var(--accent);
      color: var(--accent);
    }

    #translate-btn {
      flex: 1;
      background: linear-gradient(135deg, #1f6feb, #388bfd);
      color: #fff;
    }

    #translate-btn:hover:not([disabled]) {
      background: linear-gradient(135deg, #388bfd, #58a6ff);
      transform: translateY(-1px);
      box-shadow: 0 4px 18px rgba(56, 139, 253, 0.35);
    }

    #translate-btn[disabled] {
      opacity: 0.55;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    /* Spinner */
    @keyframes spin { to { transform: rotate(360deg); } }

    .spinner {
      display: inline-block;
      width: 13px;
      height: 13px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.65s linear infinite;
      vertical-align: middle;
      margin-right: 7px;
    }

    /* Result card */
    #result-card {
      display: none;
      text-align: center;
      padding: 32px 24px;
    }

    .result-label {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--text-muted);
      margin-bottom: 18px;
    }

    .prediction-word {
      font-size: 2.8rem;
      font-weight: 800;
      letter-spacing: -1.5px;
      background: linear-gradient(135deg, #58a6ff 0%, #a5d6ff 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      text-transform: capitalize;
      line-height: 1.1;
    }

    .latency-label {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 10px;
    }

    .latency-label span {
      color: var(--success);
      font-weight: 600;
    }

    .divider {
      height: 1px;
      background: var(--border);
      margin: 4px 0 14px;
    }
  </style>
</head>
<body>

  <header>
    <h1>ASL Sign Language Recognizer</h1>
    <p>CNN-LSTM &middot; MediaPipe Holistic &middot; 100 Classes</p>
  </header>

  <!-- Platform Card -->
  <div class="card">
    <div class="section-label">Inference Platform</div>
    <div class="platform-btns">
      <button class="platform-btn active" id="btn-ec2" type="button">
        &#9889; EC2 <small>(g5.xlarge)</small>
      </button>
      <button class="platform-btn" id="btn-sm" type="button">
        &#9729; SageMaker <small>(g5.xlarge)</small>
      </button>
    </div>
    <div class="status-badge">
      <span class="dot"></span> <span id="status-text">EC2 endpoint active</span>
    </div>
  </div>

  <!-- Upload Card -->
  <div class="card">
    <div class="section-label">Video Input</div>

    <form id="upload-form">
      <div class="drop-zone" id="drop-zone">
        <input type="file" id="videoFile" name="video" accept=".mp4,.mov,.avi" required>
        <span class="drop-icon">&#127916;</span>
        <p>Drag &amp; drop a video here, or <span class="browse-link">browse</span></p>
        <p class="formats">Accepted: .mp4 &nbsp;&middot;&nbsp; .mov &nbsp;&middot;&nbsp; .avi</p>
      </div>

      <div class="file-selected" id="file-name"></div>

      <div id="preview-section">
        <div style="height:10px;"></div>
        <video id="video-preview" controls></video>
      </div>

      <div class="btn-row">
        <button type="button" class="btn" id="preview-btn">&#9654; Preview</button>
        <button type="submit" class="btn" id="translate-btn">Translate Sign</button>
      </div>
    </form>
  </div>

  <!-- Result Card -->
  <div class="card" id="result-card">
    <div class="result-label">Prediction</div>
    <div class="divider"></div>
    <div class="prediction-word" id="prediction-text"></div>
    <div class="latency-label">Inference time: <span id="latency-text"></span></div>
    <div class="platform-label">Platform: <span id="platform-text"></span></div>
  </div>

  <script>
    var dropZone     = document.getElementById("drop-zone");
    var fileInput    = document.getElementById("videoFile");
    var fileNameEl   = document.getElementById("file-name");
    var previewBtn   = document.getElementById("preview-btn");
    var previewSec   = document.getElementById("preview-section");
    var videoEl      = document.getElementById("video-preview");
    var resultCard   = document.getElementById("result-card");
    var translateBtn = document.getElementById("translate-btn");
    var btnEC2       = document.getElementById("btn-ec2");
    var btnSM        = document.getElementById("btn-sm");
    var statusText   = document.getElementById("status-text");
    var selectedPlatform = "ec2";

    btnEC2.addEventListener("click", function() {
      selectedPlatform = "ec2";
      btnEC2.classList.add("active");
      btnSM.classList.remove("active");
      statusText.textContent = "EC2 endpoint active";
    });

    btnSM.addEventListener("click", function() {
      selectedPlatform = "sagemaker";
      btnSM.classList.add("active");
      btnEC2.classList.remove("active");
      statusText.textContent = "SageMaker endpoint active";
    });

    dropZone.addEventListener("dragover", function(e) {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", function() {
      dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", function(e) {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        onFileChosen();
      }
    });

    fileInput.addEventListener("change", onFileChosen);

    function onFileChosen() {
      var file = fileInput.files[0];
      if (!file) return;
      fileNameEl.textContent = "\\u2714 " + file.name;
      fileNameEl.style.display = "block";
      previewBtn.style.display = "inline-block";
      previewSec.style.display = "none";
      videoEl.src = "";
      resultCard.style.display = "none";
    }

    previewBtn.onclick = function() {
      var file = fileInput.files[0];
      if (!file) return;
      videoEl.src = URL.createObjectURL(file);
      previewSec.style.display = "block";
      videoEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    };

    document.getElementById("upload-form").onsubmit = async function(e) {
      e.preventDefault();

      var file = fileInput.files[0];
      if (!file) return;

      translateBtn.disabled = true;
      translateBtn.innerHTML = "<span class=\\"spinner\\"></span>Translating&hellip;";
      resultCard.style.display = "none";

      try {
        var formData = new FormData();
        formData.append("video", file);

        var endpoint = selectedPlatform === "sagemaker" ? "/predict/sagemaker" : "/predict";
        var response = await fetch(endpoint, { method: "POST", body: formData });
        var data = await response.json();

        if (data.error) {
          alert("Error: " + data.error);
          return;
        }

        if (selectedPlatform === "sagemaker") {
          document.getElementById("prediction-text").textContent = data.predicted_word;
          document.getElementById("latency-text").textContent = data.latency_ms.toFixed(1) + " ms";
          document.getElementById("platform-text").textContent = data.platform;
        } else {
          document.getElementById("prediction-text").textContent = data.prediction;
          document.getElementById("latency-text").textContent = data.latency_sec.toFixed(3) + " s";
          document.getElementById("platform-text").textContent = data.platform || "EC2";
        }

        resultCard.style.display = "block";
        resultCard.scrollIntoView({ behavior: "smooth" });
      } catch (err) {
        alert("Request failed: " + err.message);
      } finally {
        translateBtn.disabled = false;
        translateBtn.innerHTML = "Translate Sign";
      }
    };
  </script>

</body>
</html>'''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)