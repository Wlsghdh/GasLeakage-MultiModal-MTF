import torch
import torch.nn as nn
from torchvision.models import resnet50, inception_v3, mobilenet_v2

class CNNFeatureExtractor(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == 'resnet50':
            model = resnet50(weights='IMAGENET1K_V1')
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.out_dim = 2048
        elif name == 'inception':
            model = inception_v3(weights='IMAGENET1K_V1')
            model.eval()


            layers = []
            layers.append(model.Conv2d_1a_3x3)  
            layers.append(model.Conv2d_2a_3x3)   
            layers.append(model.Conv2d_2b_3x3)  
            layers.append(model.maxpool1)       
            layers.append(model.Conv2d_3b_1x1)  
            layers.append(model.Conv2d_4a_3x3)  
            layers.append(model.maxpool2)       
            layers.append(model.Mixed_5b)      
            layers.append(model.Mixed_5c)       
            layers.append(model.Mixed_5d)       
            layers.append(model.Mixed_6a)       
            
            self.features = nn.Sequential(*layers)
            self.out_dim = 768  
            
        elif name == 'mobilenet':
            model = mobilenet_v2(weights='IMAGENET1K_V1')
            self.features = model.features
            self.out_dim = 1280


    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x