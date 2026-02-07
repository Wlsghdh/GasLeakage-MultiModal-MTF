import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from src.dataset import GasDataset
from src.transforms import transform
from src.config import *
from src.models.multitask_fusion_model import MultitaskFusionModel
from src.GasDataSet import *
def train():
    # dataset = GasDataset(DATA_DIR_THERMAL, DATA_DIR_SENSOR, transform=transform)
    # train_size = int(0.8 * len(dataset))
    # train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=1)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,num_workers=1)
    
    train_dataset = GasDataSet(TRAIN_CSV_PATH, DATA_DIR_THERMAL, DATA_DIR_SENSOR, transform=transform)
    test_dataset = GasDataSet(TEST_CSV_PATH, DATA_DIR_THERMAL, DATA_DIR_SENSOR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=1)
    
    
    model = MultitaskFusionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for thermal, sensor, label in train_loader:
            thermal, sensor, label = thermal.to(DEVICE), sensor.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(thermal, sensor)
            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += (output.argmax(1) == label).sum().item()
            total += label.size(0)

        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={correct/total:.2%}")


    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for thermal, sensor, label in test_loader:
            thermal, sensor, label = thermal.to(DEVICE), sensor.to(DEVICE), label.to(DEVICE)
            output = model(thermal, sensor)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    print(f"Test Accuracy: {correct / total:.2%}")
    torch.save(model.state_dict(), 'Multitask_fusion_model.pt')
    #torch.save(test_dataset.indices, "test_indices.pt")



if __name__ == '__main__':
    train()
