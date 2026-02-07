import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, densenet121, convnext_tiny

class CNNFeatureExtractor2(nn.Module):
    def __init__(self, name):
        super().__init__()

        # ✅ EfficientNet-B0
        if name == 'efficientnet':
            model = efficientnet_b0(weights='IMAGENET1K_V1')
            self.features = model.features
            self.out_dim = 1280

        # ✅ DenseNet-121
        elif name == 'densenet':
            model = densenet121(weights='IMAGENET1K_V1')
            self.features = model.features
            self.out_dim = 1024

        # ✅ ConvNeXt-Tiny
        elif name == 'convnext':
            model = convnext_tiny(weights='IMAGENET1K_V1')
            self.features = model.features
            self.out_dim = 768

        else:
            raise ValueError(f"Unknown model name: {name}")

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x
