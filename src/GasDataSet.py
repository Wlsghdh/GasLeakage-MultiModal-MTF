import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from src.config import LABEL_MAP

class GasDataSet(Dataset):
    def __init__(self, csv_file, thermal_root, sensor_root, transform=None):

        self.df = pd.read_csv(csv_file)
        self.thermal_root = thermal_root
        self.sensor_root = sensor_root
        self.transform = transform
        
        required_columns = 'Corresponding Image Name'
  
        print(f"Dataset loaded: {len(self.df)} samples from {csv_file}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['Corresponding Image Name']+'.png'
        label_name = row['Gas']
        
        label = LABEL_MAP[label_name]
        
        thermal_path = os.path.join(self.thermal_root, label_name, image_name)
        sensor_path = os.path.join(self.sensor_root, label_name, image_name)
        
        if not os.path.exists(thermal_path):
            raise FileNotFoundError(f"Not Found Thermal_Image: {thermal_path}")
        if not os.path.exists(sensor_path):
            raise FileNotFoundError(f"Not Found Sensor_Image: {sensor_path}")
        
        thermal = Image.open(thermal_path).convert('RGB')
        sensor = Image.open(sensor_path).convert('RGB')
        
        if self.transform:
            thermal = self.transform(thermal)
            sensor = self.transform(sensor)

        return thermal, sensor, torch.tensor(label, dtype=torch.long)
