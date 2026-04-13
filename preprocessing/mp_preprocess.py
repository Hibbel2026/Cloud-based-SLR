import argparse
import cv2
from pathlib import Path
import mediapipe as mp
from multiprocessing import Pool, cpu_count

import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
DrawingSpec = mp.solutions.drawing_utils.DrawingSpec


# ===== GLOBAL (per worker) =====
holistic = None


def init_worker():
    global holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def process_frame(args):
    input_path, output_path = args

    image = cv2.imread(str(input_path))
    if image is None:
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # styles
    left = DrawingSpec(color=(255, 0, 0), thickness=2)
    right = DrawingSpec(color=(0, 0, 255), thickness=2)
    pose = DrawingSpec(color=(0, 255, 255), thickness=2)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS, pose, pose)

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS, left, left)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS, right, right)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def process_dataset(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    tasks = []

    for frame in input_root.rglob("*.jpg"):
        rel = frame.relative_to(input_root)
        out = output_root / rel
        tasks.append((frame, out))

    print(f"Total frames: {len(tasks)}")

    with Pool(cpu_count(), initializer=init_worker) as p:
        p.map(process_frame, tasks)

    print("Done!")


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("input", default="data/train", nargs="?")
    p.add_argument("output", default="data_landmark/train", nargs="?")
    args = p.parse_args()

    process_dataset(args.input, args.output)


if __name__ == "__main__":
    _cli()