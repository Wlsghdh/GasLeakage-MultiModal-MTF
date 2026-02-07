import torch
import torch.nn as nn
from src.models.extractors import CNNFeatureExtractor
from src.models.extractors2 import CNNFeatureExtractor2
from src.models.bi_lstm import BiLSTMClassifier
from src.models.utils import apply_dwt, apply_dct

class MultitaskFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_thermal = nn.ModuleList([CNNFeatureExtractor(name) for name in ['resnet50', 'inception', 'mobilenet']])
        self.cnn_sensor  = nn.ModuleList([CNNFeatureExtractor(name) for name in ['resnet50', 'inception', 'mobilenet']])
        total_dim = (2048//2 + 768//2 + 1280//2) * 2
        #total_dim = 3072
        self.bilstm = BiLSTMClassifier(input_dim=total_dim)

    def forward(self, x_thermal, x_sensor):

        thermal_features = [apply_dwt(cnn(x_thermal)) for cnn in self.cnn_thermal]
        sensor_features = [apply_dwt(cnn(x_sensor)) for cnn in self.cnn_sensor]

        f = torch.cat(thermal_features + sensor_features, dim=1)
        f = apply_dct(f)
        return self.bilstm(f)