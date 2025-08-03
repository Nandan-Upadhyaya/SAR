import os
from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from SAR_UNET_PATCHGAN import IMG_SIZE, DEVICE, TERRAIN_TYPES
from download_models import download_models
download_models()

app = Flask(__name__)

# Model classes from Final_Model_Testing.py
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
        return self.fc(terrain).unsqueeze(2).unsqueeze(3)

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
        self.backbone = models.resnet34(weights=None)
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

def load_models():
    # Load Generator
    generator = Generator().to(DEVICE)
    checkpoint = torch.load('models/gan_model.pth', map_location=DEVICE, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Load TerrainClassifier
    terrain_classifier = TerrainClassifier().to(DEVICE)
    terrain_classifier.load_state_dict(torch.load('models/classifier.pth', map_location=DEVICE, weights_only=False))
    terrain_classifier.eval()
    
    return generator, terrain_classifier

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load models globally
generator, terrain_classifier = load_models()

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and process image
        img = Image.open(file).convert('RGB')
        sar_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Get terrain prediction
        terrain_onehot, terrain_name = predict_terrain(terrain_classifier, sar_tensor)
        
        # Generate colorized image
        with torch.no_grad():
            fake = generator(sar_tensor, terrain_onehot)
        
        # Convert to numpy for display
        colorized_np = ((fake[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
        
        # Convert to PIL image and save
        colorized_uint8 = (colorized_np * 255).astype(np.uint8)
        output_image = Image.fromarray(colorized_uint8)
        
        # Create static folder if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        # Save result
        output_path = os.path.join('static', 'result.png')
        output_image.save(output_path)
        
        # Save comparison image
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Input SAR Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(colorized_np)
        plt.title(f"Colorized Image (Predicted: {terrain_name})")
        plt.axis('off')
        
        comparison_path = os.path.join('static', 'comparison.jpg')
        plt.savefig(comparison_path)
        plt.close()
        
        return jsonify({
            'success': True,
            'result_path': output_path,
            'comparison_path': comparison_path,
            'terrain_type': terrain_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)