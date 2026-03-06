import torch
from core.distiller import TGKD_Distiller

def run_distillation(teacher, student, trust_engine, loader, epochs=10, alpha=0.5, temp=2.0):
    teacher.eval() # Teacher is always frozen
    student.train()
    
    distiller = TGKD_Distiller(alpha=alpha, temperature=temp)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    
    print("🧠 Starting Trust-Gated Distillation...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass teacher (No gradients)
            with torch.no_grad():
                t_logits = teacher(features)
                # Dynamically calculate Trust Score T(x)
                trust_val = trust_engine.get_trust_score(t_logits, features, tau=temp)
            
            # Forward pass student
            s_logits = student(features)
            
            # Calculate gated loss
            loss = distiller(s_logits, t_logits, labels, trust_val)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} - TGKD Loss: {epoch_loss/len(loader):.4f}")
    return student