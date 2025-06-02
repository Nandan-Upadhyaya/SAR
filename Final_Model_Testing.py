import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------- CONFIG -------------------
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']
GENERATOR_PATH = 'checkpoints/checkpoint_epoch_139.pth'
TERRAIN_CLASSIFIER_PATH = 'terrain_classifier_checkpoints/best_terrain_classifier.pth'

# ------------------- MODEL CLASSES -------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, use_dropout=False):
        super().__init__()
        self.down = down
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        if down:
            self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.act(x)
        return x

class TerrainEncoder(nn.Module):
    def __init__(self, terrain_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(terrain_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )
    def forward(self, terrain):
        return self.fc(terrain).unsqueeze(2).unsqueeze(3)  # (B, out_dim, 1, 1)

class Generator(nn.Module):
    def __init__(self, terrain_dim=len(TERRAIN_TYPES)):
        super().__init__()
        # Encoder
        self.enc1 = UNetBlock(3, 64, down=True, use_bn=False)
        self.enc2 = UNetBlock(64, 128, down=True)
        self.enc3 = UNetBlock(128, 256, down=True)
        self.enc4 = UNetBlock(256, 512, down=True)
        # Terrain
        self.terrain_enc = TerrainEncoder(terrain_dim, 512)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.dec4 = UNetBlock(1024, 256, down=False)
        self.dec3 = UNetBlock(512, 128, down=False)
        self.dec2 = UNetBlock(256, 64, down=False)
        self.dec1 = nn.ConvTranspose2d(128, 3, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, terrain):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        t = self.terrain_enc(terrain)
        b = self.bottleneck(e4 + t)
        d4 = self.dec4(torch.cat([b, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        return self.tanh(d1)

class TerrainClassifier(nn.Module):
    def __init__(self, num_classes=len(TERRAIN_TYPES)):
        super().__init__()
        from torchvision import models
        self.backbone = models.resnet34(weights=None)  # No need to load weights here
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.dropout = nn.Dropout(0.4)
        
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

# ------------------- INFERENCE FUNCTIONS -------------------
def load_models():
    # Load Generator
    generator = Generator().to(DEVICE)
    # Fix: Use weights_only=False for legacy checkpoint compatibility
    checkpoint = torch.load(GENERATOR_PATH, map_location=DEVICE, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Load TerrainClassifier
    from torchvision import models
    terrain_classifier = TerrainClassifier().to(DEVICE)
    terrain_classifier.load_state_dict(torch.load(TERRAIN_CLASSIFIER_PATH, map_location=DEVICE, weights_only=False))
    terrain_classifier.eval()
    
    return generator, terrain_classifier

def predict_terrain(model, sar_input):
    """Predict terrain class from SAR image"""
    with torch.no_grad():
        logits = model(sar_input)
        pred_class = torch.argmax(logits, dim=1)
        # Convert to one-hot encoded terrain vector
        terrain_onehot = torch.zeros(pred_class.size(0), len(TERRAIN_TYPES), device=DEVICE)
        for i, pred in enumerate(pred_class):
            terrain_onehot[i, pred] = 1.0
        # Get terrain name for display
        terrain_name = TERRAIN_TYPES[pred_class[0].item()]
    return terrain_onehot, terrain_name

def process_image(img_path):
    """Load and preprocess the input SAR image"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    return img_tensor, img

def colorize_sar_image(img_path, output_path=None):
    """Main function to colorize a SAR image"""
    # Load models
    generator, terrain_classifier = load_models()
    
    # Process input image
    img_tensor, original_img = process_image(img_path)
    
    # Predict terrain
    terrain_onehot, terrain_name = predict_terrain(terrain_classifier, img_tensor)
    print(f"Detected terrain type: {terrain_name}")
    
    # Generate colorized image
    with torch.no_grad():
        colorized = generator(img_tensor, terrain_onehot)
    
    # Convert to numpy for display
    colorized_np = ((colorized[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    
    # Save output
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = f"{base_name}_colorized.png"
    
    # Save as PNG
    colorized_uint8 = (colorized_np * 255).astype(np.uint8)
    colorized_img = Image.fromarray(colorized_uint8)
    colorized_img.save(output_path)
    
    # Also display side by side comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title(f"Input SAR Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(colorized_np)
    plt.title(f"Colorized Image (Predicted: {terrain_name})")
    plt.axis('off')
    
    comparison_path = os.path.splitext(output_path)[0] + "_comparison.jpg"
    plt.savefig(comparison_path)
    plt.close()
    
    print(f"Colorized image saved to: {output_path}")
    print(f"Comparison image saved to: {comparison_path}")
    
    return colorized_np, terrain_name

def get_random_samples_from_terrains(dataset_root, terrains, img_size=IMG_SIZE):
    """Randomly select one SAR/Color pair from each terrain."""
    samples = []
    for terrain in terrains:
        sar_dir = os.path.join(dataset_root, terrain, 'SAR')
        color_dir = os.path.join(dataset_root, terrain, 'Color')
        sar_imgs = sorted([f for f in os.listdir(sar_dir) if not f.startswith('.')])
        color_imgs = sorted([f for f in os.listdir(color_dir) if not f.startswith('.')])
        if not sar_imgs or not color_imgs:
            continue
        idx = random.randint(0, min(len(sar_imgs), len(color_imgs)) - 1)
        sar_path = os.path.join(sar_dir, sar_imgs[idx])
        color_path = os.path.join(color_dir, color_imgs[idx])
        samples.append((terrain, sar_path, color_path))
    return samples

def preprocess_image(img_path, img_size=IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor, img

def generate_comparison_grid(generator, terrain_classifier, samples, device=DEVICE):
    sar_imgs = []
    gt_imgs = []
    gen_imgs = []
    terrain_labels = []
    for terrain, sar_path, color_path in samples:
        sar_tensor, sar_pil = preprocess_image(sar_path)
        color_tensor, color_pil = preprocess_image(color_path)
        sar_tensor = sar_tensor.unsqueeze(0).to(device)
        color_tensor = color_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            terrain_onehot, terrain_name = predict_terrain(terrain_classifier, sar_tensor)
            fake = generator(sar_tensor, terrain_onehot)
        # Denormalize for visualization
        sar_vis = ((sar_tensor[0].cpu() + 1) / 2).clamp(0, 1)
        gt_vis = ((color_tensor[0].cpu() + 1) / 2).clamp(0, 1)
        fake_vis = ((fake[0].cpu() + 1) / 2).clamp(0, 1)
        sar_imgs.append(sar_vis)
        gt_imgs.append(gt_vis)
        gen_imgs.append(fake_vis)
        terrain_labels.append(terrain)
    # Stack images: for each terrain, [SAR | GT | GEN]
    rows = []
    for i in range(len(sar_imgs)):
        row = torch.stack([sar_imgs[i], gt_imgs[i], gen_imgs[i]], dim=0)
        rows.append(row)
    # Concatenate all rows
    grid = torch.cat(rows, dim=0)
    # Make grid: 3 columns (SAR, GT, GEN), 4 rows (terrains)
    grid_img = make_grid(grid, nrow=3)
    return grid_img, terrain_labels

def save_comparison_jpg(grid_img, terrain_labels, output_path):
    np_grid = (grid_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    plt.figure(figsize=(9, 12))
    plt.imshow(np_grid)
    plt.axis('off')
    # Add terrain labels as y-ticks
    for i, terrain in enumerate(terrain_labels):
        plt.text(-10, (i * grid_img.shape[1] // 4) + 60, terrain, fontsize=14, color='white', va='center', ha='right', backgroundcolor='black')
    plt.title("Rows: Terrains | Columns: SAR, Ground Truth, Generated")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to: {output_path}")

def main_random_terrain_comparison():
    generator, terrain_classifier = load_models()
    samples = get_random_samples_from_terrains(
        dataset_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset'),
        terrains=TERRAIN_TYPES
    )
    grid_img, terrain_labels = generate_comparison_grid(generator, terrain_classifier, samples)
    save_comparison_jpg(grid_img, terrain_labels, "random_terrain_comparison.jpg")

def main_colorize_user_input():
    generator, terrain_classifier = load_models()
    img_path = input("Enter the path to the SAR image: ").strip()
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        return
    output_path = input("Enter the output path for the colorized image (leave blank for default): ").strip()
    output_path = output_path if output_path else None
    colorize_sar_image(img_path, output_path)

if __name__ == "__main__":
    # Ask user for SAR image and output colorized image
    main_colorize_user_input()
