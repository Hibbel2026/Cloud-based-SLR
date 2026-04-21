import os
import requests
import csv
import numpy as np
import time

VIDEO_DIR = "../Data_inference"
EC2_URL = "http://16.16.248.152:5000/predict"
OUTPUT_CSV = "../outputs/inference/ec2_inference_results.csv"

videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])

latencies = []
cold_start = None

total_start = time.time()

with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["video", "true_label", "prediction", "latency_sec"])

    for i, video in enumerate(videos):
        true_label = video.split("__")[0]
        video_path = os.path.join(VIDEO_DIR, video)

        with open(video_path, "rb") as f:
            response = requests.post(EC2_URL, files={"video": f})

        result = response.json()
        latency = result.get("latency_sec")
        prediction = result.get("prediction")

        if i == 0:
            cold_start = latency

        latencies.append(latency)
        writer.writerow([video, true_label, prediction, latency])
        print(f"[{i+1}/100] {true_label} → {prediction} ({latency:.2f}s)")

total_time = time.time() - total_start
latencies = np.array(latencies)

requests_per_hour = (len(videos) / total_time) * 3600
cost_per_request = 1.006 / requests_per_hour

print(f"\n===== EC2 Inference Results =====")
print(f"Cold start:        {cold_start:.4f}s")
print(f"Mean latency:      {np.mean(latencies):.4f}s")
print(f"Median latency:    {np.median(latencies):.4f}s")
print(f"Std latency:       {np.std(latencies):.4f}s")
print(f"P95 latency:       {np.percentile(latencies, 95):.4f}s")
print(f"Total time:        {total_time:.2f}s")
print(f"Requests/hour:     {requests_per_hour:.0f}")
print(f"Cost/request:      ${cost_per_request:.6f}")