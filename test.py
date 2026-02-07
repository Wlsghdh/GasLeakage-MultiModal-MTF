import torch
from torch.utils.data import DataLoader
from src.config import DATA_DIR_SENSOR, DATA_DIR_THERMAL, BATCH_SIZE, DEVICE,TEST_CSV_PATH
from src.dataset import GasDataset
from src.GasDataSet import *
from src.transforms import transform
from src.models.multitask_fusion_model import MultitaskFusionModel
from tqdm import tqdm
import time
MODEL_PATH = 'Multitask_fusion_model.pt'  
def test():
    print("Loading dataset...")
    #test_dataset = GasDataset(DATA_DIR_THERMAL, DATA_DIR_SENSOR, transform=transform)
    test_dataset = GasDataSet(TEST_CSV_PATH, DATA_DIR_THERMAL, DATA_DIR_SENSOR, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Loading model...")
    model = MultitaskFusionModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    start= time.time()

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for thermal, sensor, label in tqdm(test_loader, desc="Testing", unit="batch"):
            thermal, sensor, label = thermal.to(DEVICE), sensor.to(DEVICE), label.to(DEVICE)
            outputs = model(thermal, sensor)
            preds = outputs.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    acc = correct / total if total > 0 else 0
    print(f"[Test Accuracy] {acc * 100:.2f}%")
    print(start-time.time())
if __name__ == '__main__':
    test()
