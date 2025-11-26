import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def evaluate(model, testloader, device):
    """
    Evaluate model accuracy on test set
    """
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Update statistics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    final_acc = 100.0 * correct / total
    return final_acc


def get_cifar10_testloader(batch_size=128):
    """
    Create CIFAR-10 test data loader with proper normalization
    """
    # CIFAR-10 normalization constants (MUST match training)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    testset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return testloader


def main():
    # Set device
    device = torch.device("cpu")  # Quantized models run on CPU
    
    print("Loading quantized INT8 model...")
    
    # Load the scripted quantized model
    quant_model = torch.jit.load("vgg6_int8.pt", map_location=device)
    quant_model.eval()
    
    print("Model loaded successfully!")
    print(f"Model device: {device}")
    
    # Get test data loader
    print("\nLoading CIFAR-10 test dataset...")
    testloader = get_cifar10_testloader(batch_size=128)
    print(f"Test dataset size: {len(testloader.dataset)}")
    
    # Evaluate the quantized model
    print("\nEvaluating quantized model...")
    test_acc = evaluate(quant_model, testloader, device)
    
    print(f"\n{'='*50}")
    print(f"Quantized INT8 Model Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*50}")
    
    # Optional: Check model size
    import os
    if os.path.exists("vgg6_int8.pt"):
        model_size = os.path.getsize("vgg6_int8.pt") / (1024 * 1024)  # MB
        print(f"\nQuantized model size: {model_size:.2f} MB")
    
    # Optional: Run inference on a single batch to show it works
    print("\nRunning sample inference...")
    sample_images, sample_labels = next(iter(testloader))
    sample_images = sample_images.to(device)
    
    with torch.no_grad():
        outputs = quant_model(sample_images)
        _, predictions = outputs.max(1)
    
    print(f"Sample batch size: {sample_images.size(0)}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 10 predictions: {predictions[:10].tolist()}")
    print(f"First 10 labels: {sample_labels[:10].tolist()}")
    
    return quant_model, test_acc


if __name__ == "__main__":
    model, accuracy = main()