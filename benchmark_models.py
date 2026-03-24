import torch
import torch.nn as nn
import time
import os
import pandas as pd
import numpy as np
import argparse
from models.model import TeacherResNet, StudentMLP

def get_model_size(model_path):
    size_bin = os.path.getsize(model_path)
    return size_bin / (1024 * 1024)  # Convert to MB

def measure_performance(model, input_data, device):
    model.eval()
    model.to(device)
    
    # 1. Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    # 2. Measure Latency (Single Packet)
    single_packet = input_data[0:1]
    latencies = []
    with torch.no_grad():
        for _ in range(500):
            start_time = time.perf_counter()
            _ = model(single_packet)
            latencies.append(time.perf_counter() - start_time)
    avg_latency = np.mean(latencies) * 1000 # to ms
    
    # 3. Measure Throughput
    start_time = time.perf_counter()
    iterations = 50
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_data)
    end_time = time.perf_counter()
    
    total_packets = input_data.shape[0] * iterations
    throughput = total_packets / (end_time - start_time)

    return avg_latency, throughput

def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Benchmarking on: {device}")

    # 1. Load Checkpoints
    t_checkpoint = torch.load(args.teacher_path, map_location=device, weights_only=False)
    s_checkpoint = torch.load(args.student_path, map_location=device, weights_only=False)
    
    t_state = t_checkpoint['model_state_dict'] if 'model_state_dict' in t_checkpoint else t_checkpoint
    s_state = s_checkpoint['model_state_dict'] if 'model_state_dict' in s_checkpoint else s_checkpoint

    # 2. Dynamic Dimension Detection
    input_dim = t_state['input_layer.0.weight'].shape[1] 
    num_classes = t_state['classifier.bias'].shape[0]
    
    print(f"📊 Detected Model Dimensions: Features={input_dim}, Classes={num_classes}")

    # 3. Initialize & Load
    teacher = TeacherResNet(input_dim, num_classes).to(device)
    student = StudentMLP(input_dim, num_classes).to(device)

    teacher.load_state_dict(t_state)
    student.load_state_dict(s_state)
    
    # Dummy data for stress testing (10,000 packets)
    dummy_input = torch.randn(10000, input_dim).to(device)

    # 4. Run Benchmarking
    print("⏳ Running performance tests (Latency & Throughput)...")
    t_lat, t_thr = measure_performance(teacher, dummy_input, device)
    s_lat, s_thr = measure_performance(student, dummy_input, device)
    
    t_size = get_model_size(args.teacher_path)
    s_size = get_model_size(args.student_path)

    # 5. Build and Print Result Table
    results = {
        "Metric": ["Model Size", "Latency", "Throughput"],
        "Teacher (ResNet)": [f"{t_size:.2f} MB", f"{t_lat:.4f} ms/pkt", f"{t_thr:,.0f} pkts/sec"],
        "Student (MLP)": [f"{s_size:.2f} MB", f"{s_lat:.4f} ms/pkt", f"{s_thr:,.0f} pkts/sec"],
        "Efficiency Gain": [f"{t_size/s_size:.1f}x smaller", f"{t_lat/s_lat:.1f}x faster", f"{s_thr/t_thr:.1f}x higher"]
    }
    
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("🏆 TGKD RESEARCH BENCHMARK RESULTS")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    os.makedirs('logs', exist_ok=True)
    df.to_csv("logs/benchmark_comparison.csv", index=False)
    print("✅ Results saved to logs/benchmark_comparison.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--student_path', type=str, required=True)
    run_benchmark(parser.parse_args())