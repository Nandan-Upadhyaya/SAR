import os
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- CONFIG ---
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']
MODEL_PATH = 'terrain_classifier_checkpoints/best_terrain_classifier.pth'

# --- MODEL ---
class TerrainClassifier(torch.nn.Module):
    def __init__(self, num_classes=len(TERRAIN_TYPES)):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)
        self.dropout = torch.nn.Dropout(0.4)
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
        x = self.dropout(x)
        x = self.backbone.fc(x)
        return x

# --- IMAGE TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_terrain(model, image_path):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return TERRAIN_TYPES[pred]

if __name__ == '__main__':
    # --- Load model ---
    model = TerrainClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- Collect images from all classes ---
    dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')
    all_samples = []
    for terrain in TERRAIN_TYPES:
        sar_dir = os.path.join(dataset_root, terrain, 'SAR')
        if os.path.exists(sar_dir):
            sar_files = [os.path.join(sar_dir, f) for f in os.listdir(sar_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            for img_path in sar_files:
                all_samples.append((terrain, img_path))
    # Shuffle and pick 15 random images
    np.random.seed(42)
    np.random.shuffle(all_samples)
    selected_samples = all_samples[:15]

    # Predict and print results
    for actual, img_path in selected_samples:
        pred = predict_terrain(model, img_path)
        print(f"actual : {actual} predicted : {pred}")

# This script only loads the trained model and performs inference (prediction) on SAR images.
# There is NO training or weight update happening in this script.
# It simply predicts the terrain class for each selected SAR image and prints the results.
