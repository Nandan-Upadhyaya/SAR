import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ------------------- CONFIG -------------------
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')

# ------------------- DATASET -------------------
class SARTerrainDataset(Dataset):
    def __init__(self, root, split='train', img_size=IMG_SIZE, terrain_types=TERRAIN_TYPES):
        self.samples = []
        self.terrain_types = terrain_types
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        all_samples = []
        for terrain in terrain_types:
            sar_dir = os.path.join(root, terrain, 'SAR')
            if not os.path.exists(sar_dir):
                continue
            sar_imgs = sorted(glob.glob(os.path.join(sar_dir, '*')))
            for img_path in sar_imgs:
                all_samples.append({
                    'sar': img_path,
                    'label': terrain_types.index(terrain)
                })
        np.random.seed(42)
        np.random.shuffle(all_samples)
        n = len(all_samples)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if split == 'train':
            selected = all_samples[:n_train]
        elif split == 'val':
            selected = all_samples[n_train:n_train + n_val]
        else:
            selected = all_samples[n_train + n_val:]
        self.samples = selected

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sar_img = Image.open(sample['sar']).convert('RGB')
        sar = self.transform(sar_img)
        label = sample['label']
        return sar, label

def get_loaders(root, batch_size=BATCH_SIZE):
    train_ds = SARTerrainDataset(root, 'train')
    val_ds = SARTerrainDataset(root, 'val')
    test_ds = SARTerrainDataset(root, 'test')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
    return train_loader, val_loader, test_loader

# ------------------- MODEL -------------------
class TerrainClassifier(nn.Module):
    def __init__(self, num_classes=len(TERRAIN_TYPES)):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.dropout = nn.Dropout(0.4)  # Add dropout for regularization
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Dropout before final FC
        x = self.backbone.fc(x)
        return x

# ------------------- TRAINING -------------------
def compute_metrics(y_true, y_pred, num_classes):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1, precision, recall

def train():
    train_loader, val_loader, _ = get_loaders(DATASET_ROOT, BATCH_SIZE)
    model = TerrainClassifier().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)  # Add weight decay for regularization
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    os.makedirs('terrain_classifier_checkpoints', exist_ok=True)
    early_stop_patience = 7
    early_stop_counter = 0
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0
        y_true_train, y_pred_train = [], []
        for sar, label in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{NUM_EPOCHS}"):
            sar, label = sar.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(sar)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true_train.extend(label.cpu().numpy())
            y_pred_train.extend(preds)
        train_loss /= len(train_loader)
        train_acc, train_f1, train_prec, train_rec = compute_metrics(y_true_train, y_pred_train, len(TERRAIN_TYPES))

        # --- Validation ---
        model.eval()
        val_loss = 0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for sar, label in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{NUM_EPOCHS}"):
                sar, label = sar.to(DEVICE), label.to(DEVICE)
                logits = model(sar)
                loss = criterion(logits, label)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_true_val.extend(label.cpu().numpy())
                y_pred_val.extend(preds)
        val_loss /= len(val_loader)
        val_acc, val_f1, val_prec, val_rec = compute_metrics(y_true_val, y_pred_val, len(TERRAIN_TYPES))

        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'terrain_classifier_checkpoints/best_terrain_classifier.pth')
            print("Saved best model.")

if __name__ == '__main__':
    train()
