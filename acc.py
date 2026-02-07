
import torch
from torch.utils.data import DataLoader
from src.config import DATA_DIR_SENSOR, DATA_DIR_THERMAL, BATCH_SIZE, DEVICE, TEST_CSV_PATH
from src.dataset import GasDataset
from src.GasDataSet import *
from src.transforms import transform
from src.models.multitask_fusion_model import MultitaskFusionModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = 'Multitask_fusion_model.pt'  

def plot_confusion_matrix(cm, class_names=None, save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")

def calculate_metrics(y_true, y_pred, num_classes):
    """
    Calculate various performance metrics
    """
    # Convert to numpy arrays if they're tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'jaccard_index': jaccard,
        'confusion_matrix': cm
    }

def print_metrics(metrics):
    """
    Print formatted metrics
    """
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1-Score       : {metrics['f1_score']:.4f}")
    print(f"Jaccard Index  : {metrics['jaccard_index']:.4f}")
    print("="*50)

def print_class_wise_metrics(y_true, y_pred, class_names=None):
    """
    Print class-wise metrics
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    print("\nCLASS-WISE METRICS")
    print("="*60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    
    for i in range(len(precision)):
        class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    print("="*60)

def test():
    print("Loading dataset...")
    #test_dataset = GasDataset(DATA_DIR_THERMAL, DATA_DIR_SENSOR, transform=transform)
    #test_dataset = GasDataSet(TEST_CSV_PATH, DATA_DIR_THERMAL, DATA_DIR_SENSOR, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Loading model...")
    model = MultitaskFusionModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # Lists to store all predictions and labels
    all_preds = []
    all_labels = []
    
    print("Testing model...")
    with torch.no_grad():
        for thermal, sensor, label in tqdm(test_loader, desc="Testing", unit="batch"):
            thermal, sensor, label = thermal.to(DEVICE), sensor.to(DEVICE), label.to(DEVICE)
            outputs = model(thermal, sensor)
            preds = outputs.argmax(dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Determine number of classes
    num_classes = len(np.unique(all_labels))
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, num_classes)
    
    # Print overall metrics
    print_metrics(metrics)
    
    # Define class names (you can modify this based on your dataset)
    class_names = [f"Gas_{i}" for i in range(num_classes)]
    # If you know the actual class names, replace the above line with:
    # class_names = ['No Gas', 'Methane', 'Ethane', 'Propane', ...]  # example
    
    # Print class-wise metrics
    print_class_wise_metrics(all_labels, all_preds, class_names)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # Print confusion matrix in text format
    print("\nCONFUSION MATRIX")
    print("="*50)
    print("Rows: True Labels, Columns: Predicted Labels")
    print(metrics['confusion_matrix'])
    print("="*50)
    
    # Additional analysis
    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for class_idx, count in zip(unique, counts):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        print(f"  {class_name}: {count} ({count/len(all_labels)*100:.1f}%)")

if __name__ == '__main__':
    test()