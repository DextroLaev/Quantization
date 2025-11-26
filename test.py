# test.py
"""
Simple testing script to load model and print test accuracy.
"""

import torch
from data_loader import get_cifar10
from models import get_vgg_model

def evaluate(model, testloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((testloader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        final_acc = 100 * correct / total
    return final_acc

def main():
    # Configuration
    MODEL_NAME = 'vgg6'
    MODEL_PATH = './models/vgg6_cifar10.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = get_vgg_model(MODEL_NAME, num_classes=10, batch_norm=True, activation='gelu').to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load test data
    print("Loading test data...")
    train_loader, test_loader = get_cifar10(batch_size=64, num_workers=8)
    
    # Evaluate
    print("Evaluating...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_acc = 100.0 * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()