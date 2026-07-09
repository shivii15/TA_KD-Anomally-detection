import argparse
import os
import json
from datetime import datetime

import torch
import torch.nn as nn

from data.data_loader import load_dataset, get_dataloaders

from models.model import (
    TeacherDNN,
    TeacherResNet,
    TeacherTransformer,
    TeacherLSTM,
    StudentMLP
)
def parse_args():

    parser = argparse.ArgumentParser(
        description="TGKD Evaluation Pipeline"
    )

    parser.add_argument(
        "--dataset",
        choices=["cic", "nbaiot"],
        required=True
    )

    parser.add_argument(
        "--data_path",
        default=None,
        type=str
    )

    parser.add_argument(
        "--model",
        choices=[
            "student",
            "dnn",
            "resnet",
            "transformer",
            "lstm"
        ],
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str
    )

    parser.add_argument(
        "--batch_size",
        default=2048,
        type=int
    )

    parser.add_argument(
        "--num_parts",
        default=-1,
        type=int
    )

    return parser.parse_args()

def main():

    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nDevice : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Model : {args.model}")
    print(f"Checkpoint : {args.checkpoint}")

    X, y, le, scaler = load_dataset(args)
    _, test_loader = get_dataloaders(
        X,
        y,
        batch_size=args.batch_size
    )

    input_dim = X.shape[1]
    num_classes = len(le.classes_)

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Samples : {len(X):,}")
    print(f"Features : {input_dim}")
    print(f"Classes : {num_classes}")
    print("-" * 40)

    model_map = {
        "dnn": TeacherDNN,
        "resnet": TeacherResNet,
        "transformer": TeacherTransformer,
        "lstm": TeacherLSTM,
        "student": StudentMLP
    }

    model = model_map[args.model](
        input_dim,
        num_classes
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )
    print("\nCheckpoint Contents")
    print("-" * 40)

    for key in checkpoint.keys():
        print(key)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nCheckpoint Loaded Successfully")
    print("-" * 40)
    print(f"Model : {args.model}")
    print(f"Dataset : {args.dataset}")
    print(f"Classes : {num_classes}")
    print("-" * 40)

if __name__ == "__main__":
    main()



