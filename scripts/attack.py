def fgsm_attack(data, epsilon, data_grad):
    # Create the perturbed image/packet
    perturbed_data = data + epsilon * data_grad.sign()
    return torch.clamp(perturbed_data, 0, 1)

def test_robustness(student, teacher, test_loader, epsilon, device):
    # Subject Teacher to attack, then check Student accuracy
    # This proves the Trust Gate blocks the 'Knowledge Poisoning'
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        
        outputs, _ = teacher(x)
        loss = F.cross_entropy(outputs, y)
        teacher.zero_grad()
        loss.backward()
        
        # Generate Adversarial Data
        x_adv = fgsm_attack(x, epsilon, x.grad.data)
        
        # Test Student on the "Poisoned" advice
        with torch.no_grad():
            s_logits, _ = student(x_adv)
            _, predicted = s_logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    return 100. * correct / total