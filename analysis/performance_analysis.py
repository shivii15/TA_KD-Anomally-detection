import os
import json
import time
import argparse

import torch
import numpy as np

from data.data_loader import load_dataset
from data.common import get_dataloaders
from models.student import StudentMLP

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--dataset",
        default="nbaiot"
    )

    parser.add_argument(
        "--data_path",
        default=None
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device : {device}")

    X, y, le, scaler = load_dataset(args)

    _, _, test_loader = get_dataloaders(
        X,
        y,
        batch_size=args.batch_size
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model = StudentMLP(
        checkpoint["input_dim"],
        checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    model_size = os.path.getsize(
        args.checkpoint
    ) / (1024**2)


    start = time.perf_counter()
    samples = 0
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            model(batch_x)
            samples += batch_x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    throughput = samples / total_time
    latency = total_time / samples * 1000

    if device.type == "cuda":

        memory = (
            torch.cuda.max_memory_allocated()
            /1024**2
        )

    else:

        memory = 0

    print()
    print("="*60)
    print("Performance Summary")
    print("="*60)
    print(f"Samples           : {samples}")
    print(f"Time (s)          : {total_time:.4f}")
    print(f"Throughput        : {throughput:.2f} samples/sec")
    print(f"Latency           : {latency:.4f} ms/sample")
    print(f"Parameters        : {trainable_params:,}")
    print(f"Model Size        : {model_size:.2f} MB")
    print(f"Peak GPU Memory   : {memory:.2f} MB")
    print("="*60)

    performance = {
        "samples": int(samples),
        "throughput": float(throughput),
        "latency_ms": float(latency),
        "parameters": int(trainable_params),
        "model_size_mb": float(model_size),
        "gpu_memory_mb": float(memory)
    }
    analysis_dir = os.path.join(
        os.path.dirname(args.checkpoint),
        "analysis"
    )
    os.makedirs(
        analysis_dir,
        exist_ok=True
    )
    with open(
        os.path.join(
            analysis_dir,
            "performance.json"
        ),
        "w"
    ) as f:

        json.dump(
            performance,
            f,
            indent=4
        )

if __name__ == "__main__":
    main()









