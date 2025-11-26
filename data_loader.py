# data_loader.py
"""
Data loading utilities for CIFAR10 dataset.
Handles dataset preparation with appropriate transforms.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_augmentation import Cutout


def get_cifar10(batch_size=64, num_workers=8, use_augmentation=False):
    """
    Load CIFAR10 dataset with train and test splits.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker threads for data loading
        use_augmentation (bool): Whether to apply AutoAugment
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Normalization constants for CIFAR10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(n_holes=1, length=16)
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        './data',
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = datasets.CIFAR10(
        './data',
        train=False,
        transform=test_transform,
        download=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader