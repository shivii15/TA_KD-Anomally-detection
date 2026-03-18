import torch
import time
import os
from models.model import TeacherResNet, StudentMLP

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, input_batch, device, repetitions=100):
    model.eval()
    model.to(device)
    input_batch = input_batch.to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_batch)
    
    # Start timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(input_batch)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / repetitions
    return avg_latency * 1000  # Convert to milliseconds

def evaluate_efficiency(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, args.input_dim) # Single sample for edge latency

    # 1. Initialize Models
    teacher = TeacherResNet(args.input_dim, args.num_classes)
    student = StudentMLP(args.input_dim, args.num_classes)

    # 2. Measure Parameters
    t_params = count_parameters(teacher)
    s_params = count_parameters(student)

    # 3. Measure Latency
    t_latency = measure_latency(teacher, dummy_input, device)
    s_latency = measure_latency(student, dummy_input, device)

    # 4. Measure Model Size
    torch.save(teacher.state_dict(), "temp_t.pth")
    torch.save(student.state_dict(), "temp_s.pth")
    t_size = os.path.getsize("temp_t.pth") / (1024 * 1024)
    s_size = os.path.getsize("temp_s.pth") / (1024 * 1024)
    os.remove("temp_t.pth")
    os.remove("temp_s.pth")

    print("\n" + "="*40)
    print(f"{'Metric':<20} | {'Teacher':<10} | {'Student':<10}")
    print("-" * 45)
    print(f"{'Parameters':<20} | {t_params:<10} | {s_params:<10}")
    print(f"{'Model Size (MB)':<20} | {t_size:<10.4f} | {s_size:<10.4f}")
    print(f"{'Latency (ms/req)':<20} | {t_latency:<10.4f} | {s_latency:<10.4f}")
    print("="*40)
    
    reduction = (t_params / s_params)
    speedup = (t_latency / s_latency)
    print(f"🚀 Efficiency Gain: {reduction:.1f}x fewer params")
    print(f"🚀 Speed Gain: {speedup:.1f}x faster inference")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=77) # CIC-IoT-2023 feature count
    parser.add_argument('--num_classes', type=int, default=34)
    args = parser.parse_args()
    evaluate_efficiency(args)