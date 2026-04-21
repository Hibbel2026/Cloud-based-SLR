import os
import json
import torch
import numpy as np

from cnn_lstm import CNN_LSTM

# ===== CLASS LABELS (100 ASL words) =====
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

NUM_CLASSES = 100


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM(num_classes=NUM_CLASSES).to(device)
    model_path = os.path.join(model_dir, "best_model_2_1774971874.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path} on {device}")
    return model


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(request_body)
    tensor = torch.tensor(data, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tensor.to(device)


def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


def output_fn(prediction, accept):
    predicted_word = CLASSES[prediction]
    result = {
        "predicted_word": predicted_word,
        "class_index": prediction
    }
    return json.dumps(result), "application/json"
