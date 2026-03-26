import torch
import time
import os
from models.model import StudentMLP, TeacherResNet, TeacherTransformer, TeacherLSTM

def get_model_size(file_path):
    size = os.path.getsize(file_path) / (1024 * 1024) # MB
    return size

def measure_latency(model, input_data, device):
    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # Warm up
    for _ in range(10):
        _ = model(input_data)
    
    with torch.no_grad():
        starter.record()
        for _ in range(100):
            _ = model(input_data)
        ender.record()
        torch.cuda.synchronize()
    
    return starter.elapsed_time(ender) / 100 # Average ms per batch

def benchmark():
    device = torch.device("cuda")
    dummy_input = torch.randn(1, 46).to(device) # Batch size 1 for real-time latency
    
    # Define models (ensure these classes are imported)
    models = {
        "ResNet Teacher": ("models/teachers/resnet_best.pth", TeacherResNet(46, 34)),
        "Student (Ours)": ("models/student_tgkd_best.pth", StudentMLP(46, 34))
    }

    print(f"{'Model':<20} | {'Size (MB)':<10} | {'Latency (ms)':<15}")
    print("-" * 50)

    for name, (path, model) in models.items():
        model.to(device)
        size = get_model_size(path)
        latency = measure_latency(model, dummy_input, device)
        print(f"{name:<20} | {size:<10.2f} | {latency:<15.4f}")

if __name__ == "__main__":
    benchmark()