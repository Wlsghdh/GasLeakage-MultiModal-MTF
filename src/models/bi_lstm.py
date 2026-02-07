import torch.nn as nn
from src.config import NUM_CLASSES


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim*2, NUM_CLASSES)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  
        
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.classifier(out)