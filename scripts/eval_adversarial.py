import torch
import torch.nn as nn

def fgsm_attack(features, epsilon, data_grad):
    """
    Generates adversarial IoT features by moving in the direction of the gradient sign.
    """
    perturbed_features = features + epsilon * data_grad.sign()
    # Ensure features remain within valid normalized range if necessary
    return perturbed_features

def evaluate_robustness(teacher, student, loader, trust_engine, epsilon=0.1):
    teacher.eval()
    student.eval()
    
    correct_tgkd = 0
    total = 0
    
    print(f"🛡️ Running Adversarial Test with Epsilon: {epsilon}")

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        features.requires_grad = True

        # 1. Target the Teacher to find its weak spots
        outputs_t = teacher(features)
        loss = nn.CrossEntropyLoss()(outputs_t, labels)
        teacher.zero_grad()
        loss.backward()

        # 2. Create Perturbed "Adversarial" IoT Traffic
        adv_features = fgsm_attack(features, epsilon, features.grad)

        # 3. Use the Trust Engine to check the attack
        with torch.no_grad():
            t_logits_adv = teacher(adv_features)
            # T(x) should drop significantly for adv_features
            T_x = trust_engine.get_trust_score(t_logits_adv, adv_features)
            
            # 4. Student Prediction on Attack Data
            s_logits = student(adv_features)
            _, predicted = torch.max(s_logits.data, 1)
            
            total += labels.size(0)
            correct_tgkd += (predicted == labels).sum().item()

    accuracy = 100 * correct_tgkd / total
    return accuracy