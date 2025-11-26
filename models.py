# models.py
"""
VGG model architecture and configurations.
Provides flexible VGG implementations with different depths.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math


class VGG(nn.Module):
    """
    VGG network architecture.
    
    Args:
        features (nn.Module): Feature extraction layers
        num_classes (int): Number of output classes
    """
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize model weights using proper initialization techniques."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, activation='gelu'):
    """
    Construct feature layers from config.
    
    Args:
        cfg (list): Configuration list with layer specs
        batch_norm (bool): Whether to include batch normalization
        activation (str): Activation function ('gelu' or 'relu')
        
    Returns:
        nn.Sequential: Feature extraction layers
    """
    layers = []
    in_channels = 3
    
    # Select activation function
    if activation.lower() == 'gelu':
        act_fn = nn.GELU()
    elif activation.lower() == 'relu':
        act_fn = nn.ReLU(inplace=True)
    else:
        raise ValueError(f"Unknown activation: {activation}. Use 'gelu' or 'relu'")
    
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(v), act_fn])
            else:
                layers.extend([conv2d, act_fn])
            in_channels = v
    
    return nn.Sequential(*layers)


def create_vgg(cfg, num_classes=10, batch_norm=True, activation='gelu'):
    """
    Create a VGG model.
    
    Args:
        cfg (list): Configuration list for layers
        num_classes (int): Number of output classes
        batch_norm (bool): Whether to use batch normalization
        activation (str): Activation function ('gelu' or 'relu')
        
    Returns:
        VGG: VGG model instance
    """
    return VGG(make_layers(cfg, batch_norm=batch_norm, activation=activation), num_classes=num_classes)


# VGG configurations
VGG_CONFIGS = {
    'vgg6': [64, 64, 'M', 128, 128, 'M'],
    'vgg8': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def get_vgg_model(model_name='vgg6', num_classes=10, batch_norm=True, activation='gelu'):
    """
    Get a VGG model by name.
    
    Args:
        model_name (str): Name of VGG model (vgg6, vgg8, etc.)
        num_classes (int): Number of output classes
        batch_norm (bool): Whether to use batch normalization
        activation (str): Activation function ('gelu' or 'relu')
        
    Returns:
        VGG: VGG model instance
    """
    if model_name not in VGG_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VGG_CONFIGS.keys())}")
    
    cfg = VGG_CONFIGS[model_name]
    return create_vgg(cfg, num_classes=num_classes, batch_norm=batch_norm, activation=activation)