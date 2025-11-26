# main.py
"""
Main training script. Orchestrates data loading, model creation, and training.
Easily configurable for experimenting with different architectures and hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_cifar10
from models import get_vgg_model
from trainer import Trainer


def main():
    """Main training pipeline."""
    
    # ============= Configuration =============
    MODEL_NAME = 'vgg6'  # Options: vgg6, vgg8, vgg11, vgg13, vgg16, vgg19
    BATCH_SIZE = 256
    NUM_WORKERS = 8
    NUM_CLASSES = 10
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    OPTIMIZER_TYPE = 'adam'  # Options: adam, sgd, rmsprop
    USE_BATCH_NORM = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    
    # ============= Load Data =============
    print("\nLoading CIFAR10 dataset...")
    train_loader, test_loader = get_cifar10(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ============= Create Model =============
    print(f"\nCreating {MODEL_NAME} model...")
    model = get_vgg_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        batch_norm=USE_BATCH_NORM
    ).to(DEVICE)
    print(model)
    
    # ============= Setup Training =============
    criterion = nn.CrossEntropyLoss()
    
    if OPTIMIZER_TYPE.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_TYPE.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    elif OPTIMIZER_TYPE.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER_TYPE}")
    
    # ============= Train Model =============
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Optimizer: {OPTIMIZER_TYPE}, Learning Rate: {LEARNING_RATE}")
    
    trainer = Trainer(
        model=model,
        device=DEVICE,
        criterion=criterion,
        optimizer=optimizer
    )
    
    trainer.train(
        epochs=NUM_EPOCHS,
        train_loader=train_loader,
        test_loader=test_loader,
        save_path='./models/vgg6_cifar10.pth',
        verbose=True
    )
    
    # ============= Results =============
    metrics = trainer.get_metrics()
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Best Train Accuracy: {max(metrics['train_accuracies']):.2f}%")
    print(f"Best Test Accuracy: {trainer.best_test_acc:.2f}% (Epoch {trainer.best_epoch})")
    print(f"Final Test Accuracy: {metrics['test_accuracies'][-1]:.2f}%")
    
    # ============= Save Model =============
    model_path = f"./models/{MODEL_NAME}_cifar10.pth"
    print(f"\nBest model saved to {model_path}")
    
    return trainer, metrics


if __name__ == "__main__":
    trainer, metrics = main()