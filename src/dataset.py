import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from src.config import LABEL_MAP

class GasDataset(Dataset):
    def __init__(self, thermal_root, sensor_root, transform=None):
        self.samples = []
        self.transform = transform

        for label_name, label in LABEL_MAP.items():
            thermal_dir = os.path.join(thermal_root, label_name)
            sensor_dir = os.path.join(sensor_root, label_name)

            if not os.path.exists(thermal_dir) or not os.path.exists(sensor_dir):
                print(f"Directory not found - {thermal_dir} or {sensor_dir}")
                continue

            thermal_imgs = sorted([f for f in os.listdir(thermal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            sensor_imgs = sorted([f for f in os.listdir(sensor_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            min_len = min(len(thermal_imgs), len(sensor_imgs)) 

            
            for i in range(min_len):
                t_path = os.path.join(thermal_dir, thermal_imgs[i])
                s_path = os.path.join(sensor_dir, sensor_imgs[i])
                self.samples.append((t_path, s_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_path, s_path, label = self.samples[idx]
        
        thermal = Image.open(t_path).convert('RGB')
        sensor = Image.open(s_path).convert('RGB')
        

        if self.transform:
            thermal = self.transform(thermal)
            sensor = self.transform(sensor)

        return thermal, sensor, torch.tensor(label, dtype=torch.long)
        
