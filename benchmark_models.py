import torch
import time
import os
import pandas as pd
from models.model import TeacherResNet, StudentMLP

def get_model_size(model_path):
    size_bin = os.path.getsize(model_path)
    return size_bin / (1024 * 1024)  # Convert to MB

def measure_performance(model, input_data, device, label="Model"):
    model.eval()
    model.to(device)
    
    # 1. Warm-up (Standard practice for GPU/CPU benchmarking)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    # 2. Measure Latency (Single Packet)
    single_packet = input_data[0:1]
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(1000):
            _ = model(single_packet)
    latency = (time.perf_counter() - start_time) / 1000 * 1000 # in ms
    
    # 3. Measure Throughput (Batch Processing)
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_data)
    end_time = time.perf_counter()
    total_packets = input_data.shape[0] * 100
    throughput = total_packets / (end_time - start_time)

    return latency, throughput

def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Checkpoints First to detect dimensions
    t_checkpoint = torch.load(args.teacher_path, map_location=device, weights_only=False)
    s_checkpoint = torch.load(args.student_path, map_location=device, weights_only=False)
    
    # Extract weights to find shapes
    t_state = t_checkpoint['model_state_dict'] if 'model_state_dict' in t_checkpoint else t_checkpoint
    s_state = s_checkpoint['model_state_dict'] if 'model_state_dict' in s_checkpoint else s_checkpoint

    # --- DYNAMIC DIMENSION DETECTION ---
    # Look at the first layer's weight to find input features (dim 1)
    input_dim = t_state['input_layer.0.weight'].shape[1] 
    
    # Look at the classifier bias to find number of classes
    num_classes = t_state['classifier.bias'].shape[0]
    
    print(f"📊 Detected Dimensions: Features={input_dim}, Classes={num_classes}")

    # 2. Initialize Models with correct dimensions
    teacher = TeacherResNet(input_dim, num_classes)
    student = StudentMLP(input_dim, num_classes)

    # 3. Load State Dicts
    teacher.load_state_dict(t_state)
    student.load_state_dict(s_state)
    
    # Create dummy data matching the DETECTED input_dim
    dummy_input = torch.randn(10000, input_dim).to(device)

    # 4. Run Benchmarking
    t_lat, t_thr = measure_performance(teacher, dummy_input, device)
    s_lat, s_thr = measure_performance(student, dummy_input, device)
    
    # ... (rest of the size and print logic remains the same)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--student_path', type=str, required=True)
    run_benchmark(parser.parse_args())