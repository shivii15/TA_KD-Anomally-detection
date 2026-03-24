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
    input_dim = 77  # Standard for CIC-IoT-2023
    num_classes = 33 
    dummy_input = torch.randn(10000, input_dim).to(device)

    # Load Teacher
    teacher = TeacherResNet(input_dim, num_classes)
    t_checkpoint = torch.load(args.teacher_path, map_location=device, weights_only=False)
    teacher.load_state_dict(t_checkpoint['model_state_dict'] if 'model_state_dict' in t_checkpoint else t_checkpoint)
    
    # Load Student
    student = StudentMLP(input_dim, num_classes)
    s_checkpoint = torch.load(args.student_path, map_location=device, weights_only=False)
    student.load_state_dict(s_checkpoint['model_state_dict'])

    # Benchmarking
    t_lat, t_thr = measure_performance(teacher, dummy_input, device)
    s_lat, s_thr = measure_performance(student, dummy_input, device)
    
    t_size = get_model_size(args.teacher_path)
    s_size = get_model_size(args.student_path)

    # Results Table
    results = {
        "Metric": ["Model Size (MB)", "Latency (ms/pkt)", "Throughput (pkts/sec)"],
        "Teacher (ResNet)": [f"{t_size:.2f}", f"{t_lat:.4f}", f"{t_thr:,.0f}"],
        "Student (MLP)": [f"{s_size:.2f}", f"{s_lat:.4f}", f"{s_thr:,.0f}"],
        "Improvement": [f"{t_size/s_size:.1f}x smaller", f"{t_lat/s_lat:.1f}x faster", f"{s_thr/t_thr:.1f}x higher"]
    }
    
    df = pd.DataFrame(results)
    print("\n🚀 --- TGKD BENCHMARK RESULTS ---")
    print(df.to_string(index=False))
    df.to_csv("logs/benchmark_results.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--student_path', type=str, required=True)
    run_benchmark(parser.parse_args())