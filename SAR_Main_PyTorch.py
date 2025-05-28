import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import autocast, GradScaler  # Updated import path
import torchvision
from torchvision import transforms, models
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
from collections import defaultdict
from scipy.linalg import sqrtm
from PIL import Image

# Configure GPU settings
def setup_devices():
    """Set up devices based on availability, optimized for Kaggle's 2 GPUs"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} GPUs")
        # Use cuda:0 as primary device
        device = torch.device("cuda:0")
        
        # Configure memory growth for each GPU
        for i in range(device_count):
            try:
                # Set memory fraction for better memory management
                torch.cuda.set_per_process_memory_fraction(0.85, i)
                # Clear GPU cache
                torch.cuda.empty_cache()
                # Print device properties for debugging
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.2f} GB")
            except Exception as e:
                print(f"Warning setting GPU {i} memory config: {e}")
    else:
        device_count = 0
        device = torch.device("cpu")
        print("No GPUs found, using CPU")
    
    return device, device_count

# Get device information
device, num_gpus = setup_devices()

# Constants (same as original)
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']
GRADIENT_ACCUMULATION_STEPS = 4

# PyTorch equivalent of InstanceNormalization
class InstanceNorm(nn.Module):
    """Instance Normalization for PyTorch"""
    def __init__(self, channels, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(channels))
        self.offset = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.var(x, dim=(2, 3), keepdim=True, unbiased=False)
        normalized = (x - mean) / (torch.sqrt(var + self.epsilon))
        return self.scale.view(1, -1, 1, 1) * normalized + self.offset.view(1, -1, 1, 1)

# Feature extractor for perceptual loss
def create_feature_extractor():
    """Create feature extractor using EfficientNetB0"""
    model = models.efficientnet_b0(pretrained=True)
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Select specific layers for feature extraction
    layers = ['features.2', 'features.4', 'features.6']
    
    class FeatureExtractor(nn.Module):
        def __init__(self, model, layers):
            super().__init__()
            self.model = model
            self.layers = layers
            self._features = {layer: torch.empty(0) for layer in layers}
            
            # Register hooks to extract features
            for layer_id in layers:
                layer = dict([*self.model.named_modules()])[layer_id]
                layer.register_forward_hook(self.save_output_hook(layer_id))
        
        def save_output_hook(self, layer_id):
            def hook(module, input, output):
                self._features[layer_id] = output
            return hook
        
        def forward(self, x):
            _ = self.model(x)
            return [self._features[layer_id] for layer_id in self.layers]
    
    return FeatureExtractor(model, layers).to(device).eval()

# Initialize feature extractor
feature_extractor = create_feature_extractor()

# Compute perceptual loss using PyTorch
def compute_perceptual_loss(real_images, generated_images):
    """Compute perceptual loss between real and generated images"""
    # Normalize images for EfficientNet
    real_images = (real_images + 1) * 127.5
    real_images = torchvision.transforms.functional.normalize(
        real_images, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    generated_images = (generated_images + 1) * 127.5
    generated_images = torchvision.transforms.functional.normalize(
        generated_images, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    with torch.no_grad():
        real_features = feature_extractor(real_images)
        gen_features = feature_extractor(generated_images)
    
    perceptual_loss = 0.0
    for real_feat, gen_feat in zip(real_features, gen_features):
        perceptual_loss += torch.mean(torch.abs(real_feat - gen_feat))
    
    return perceptual_loss

# Metrics tracker class
class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)
    
    def update(self, name, value, count=1):
        self.metrics[name] += value * count
        self.counts[name] += count
    
    def average(self, name):
        if self.counts[name] == 0:
            return 0.0
        return self.metrics[name] / self.counts[name]
    
    def reset(self):
        self.metrics.clear()
        self.counts.clear()

# Dataset class for SAR images
class SARDataset(Dataset):
    def __init__(self, sar_files, color_files, terrain_label):
        self.sar_files = sar_files
        self.color_files = color_files
        self.terrain_label = terrain_label
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.sar_files)
    
    def __getitem__(self, idx):
        sar_img = Image.open(self.sar_files[idx]).convert('RGB')
        color_img = Image.open(self.color_files[idx]).convert('RGB')
        
        sar_tensor = self.transform(sar_img)
        color_tensor = self.transform(color_img)
        
        return sar_tensor, color_tensor, self.terrain_label

def create_dataset():
    """Create PyTorch DataLoaders for training, validation and testing"""
    all_datasets = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for terrain_idx, terrain in enumerate(TERRAIN_TYPES):
        # One-hot encoded terrain labels
        terrain_label = torch.zeros(len(TERRAIN_TYPES))
        terrain_label[terrain_idx] = 1.0
        
        # Paths to SAR and color images
        sar_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s1/*'
        color_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s2/*'
        
        sar_files = sorted(glob(sar_path))
        color_files = sorted(glob(color_path))
        
        # Ensure same number of files
        min_files = min(len(sar_files), len(color_files))
        sar_files = sar_files[:min_files]
        color_files = color_files[:min_files]
        
        # Verify pairing
        for sar, color in zip(sar_files[:5], color_files[:5]):
            sar_id = os.path.basename(sar).split('_')[1].split('.')[0]
            color_id = os.path.basename(color).split('_')[1].split('.')[0]
            if sar_id != color_id:
                raise ValueError(f"Mismatched pair: {sar} vs {color}")
        
        # Calculate split indices
        total_files = len(sar_files)
        train_idx = int(total_files * 0.75)
        val_idx = int(total_files * 0.90)
        
        # Create train/val/test splits
        sar_train = sar_files[:train_idx]
        sar_val = sar_files[train_idx:val_idx]
        sar_test = sar_files[val_idx:]
        
        color_train = color_files[:train_idx]
        color_val = color_files[train_idx:val_idx]
        color_test = color_files[val_idx:]
        
        print(f"Terrain {terrain}: {len(sar_train)} train, {len(sar_val)} validation, {len(sar_test)} test")
        
        # Create datasets
        train_dataset = SARDataset(sar_train, color_train, terrain_label)
        val_dataset = SARDataset(sar_val, color_val, terrain_label)
        test_dataset = SARDataset(sar_test, color_test, terrain_label)
        
        all_datasets['train'].append(train_dataset)
        all_datasets['val'].append(val_dataset)
        all_datasets['test'].append(test_dataset)
    
    # Concatenate datasets for each split
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(all_datasets['train'])
    val_dataset = ConcatDataset(all_datasets['val'])
    test_dataset = ConcatDataset(all_datasets['test'])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE * max(1, num_gpus),
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * max(1, num_gpus),
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * max(1, num_gpus),
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader

# PyTorch equivalent of TerrainGuidedAttention
class TerrainGuidedAttention(nn.Module):
    def __init__(self, channels):
        super(TerrainGuidedAttention, self).__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Fixed input size to match terrain feature size
        self.terrain_input_dim = len(TERRAIN_TYPES)
        self.terrain_project = nn.Linear(self.terrain_input_dim, channels)
    
    def forward(self, inputs):
        x, terrain = inputs
        batch_size, _, height, width = x.shape
        
        # Ensure terrain has correct shape [B, terrain_input_dim]
        if len(terrain.shape) > 2:
            # Reshape if needed (happens with DataParallel sometimes)
            terrain = terrain.view(-1, self.terrain_input_dim)
        
        # Project terrain features
        terrain_features = self.terrain_project(terrain)  # [B, C]
        terrain_features = terrain_features.view(-1, self.channels, 1, 1)
        terrain_features = terrain_features.expand(-1, -1, height, width)
        
        # Add terrain context
        x = x + terrain_features
        
        # Compute attention
        q = self.query(x)  # [B, C/8, H, W]
        k = self.key(x)    # [B, C/8, H, W]
        v = self.value(x)  # [B, C, H, W]
        
        # Reshape for matrix multiplication
        q_flat = q.view(batch_size, -1, height * width)  # [B, C/8, H*W]
        k_flat = k.view(batch_size, -1, height * width)  # [B, C/8, H*W]
        v_flat = v.view(batch_size, -1, height * width)  # [B, C, H*W]
        
        # Calculate attention scores
        attention = torch.bmm(q_flat.permute(0, 2, 1), k_flat)  # [B, H*W, H*W]
        attention = F.softmax(attention / (self.channels ** 0.5), dim=2)
        
        # Apply attention and reshape
        out = torch.bmm(v_flat, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(batch_size, self.channels, height, width)  # [B, C, H, W]
        
        return self.gamma * out + x

# PyTorch equivalent of TerrainAdaptiveNormalization
class TerrainAdaptiveNormalization(nn.Module):
    def __init__(self, channels):
        super(TerrainAdaptiveNormalization, self).__init__()
        self.channels = channels
        self.norm = nn.BatchNorm2d(channels)
        
        # Fixed input size to match terrain feature size (always 4 terrain types)
        self.terrain_input_dim = len(TERRAIN_TYPES)
        
        # Correctly define linear transformations with proper input/output sizes
        self.terrain_scale = nn.Linear(self.terrain_input_dim, channels)
        self.terrain_bias = nn.Linear(self.terrain_input_dim, channels)
        self.color_norm = nn.InstanceNorm2d(channels, affine=False)
        self.color_scale = nn.Linear(self.terrain_input_dim, channels)
        self.color_bias = nn.Linear(self.terrain_input_dim, channels)
    
    def forward(self, inputs, training=True):
        x, terrain_features = inputs
        
        # Apply batch normalization
        normalized = self.norm(x)
        
        # Generate terrain-specific parameters
        # Ensure terrain_features has correct shape [B, terrain_input_dim]
        if len(terrain_features.shape) > 2:
            # Reshape if needed (happens with DataParallel sometimes)
            terrain_features = terrain_features.view(-1, self.terrain_input_dim)
        
        scale = self.terrain_scale(terrain_features).view(-1, self.channels, 1, 1)
        bias = self.terrain_bias(terrain_features).view(-1, self.channels, 1, 1)
        
        # Apply color normalization
        color_normalized = self.color_norm(normalized)
        color_scale = self.color_scale(terrain_features).view(-1, self.channels, 1, 1)
        color_bias = self.color_bias(terrain_features).view(-1, self.channels, 1, 1)
        
        # Combine normalizations
        terrain_norm = normalized * (1 + scale) + bias
        color_norm = color_normalized * (1 + color_scale) + color_bias
        
        return (terrain_norm + color_norm) / 2

# PyTorch equivalent of MemoryEfficientResBlock
class MemoryEfficientResBlock(nn.Module):
    def __init__(self, filters):
        super(MemoryEfficientResBlock, self).__init__()
        self.filters = filters
        self.input_proj = None
        
        # Conv layers
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=1)
        
        # Normalization layers
        self.norm1 = InstanceNorm(filters)
        self.norm2 = InstanceNorm(filters)
        
        # Activation
        self.activation = nn.SiLU()
        
        # Attention layer
        self.attention = TerrainGuidedAttention(filters)
    
    def forward(self, inputs):
        x, terrain = inputs
        
        # Handle potential input projection
        if self.input_proj is None and x.size(1) != self.filters:
            self.input_proj = nn.Conv2d(x.size(1), self.filters, kernel_size=1, padding=0).to(x.device)
        
        # Save residual connection
        residual = x
        if self.input_proj is not None:
            x = self.input_proj(x)
            residual = x
        
        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Apply attention
        x = self.attention([x, terrain])
        
        # Add residual
        x = x + residual
        
        return x

# PyTorch equivalent of SpectralNormalization
def spectral_norm(module):
    """Apply spectral normalization to a module"""
    return nn.utils.spectral_norm(module)
        
# Generator model in PyTorch
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Initial processing without downsampling
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            InstanceNorm(64),
            nn.SiLU()
        )
        
        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            InstanceNorm(128),
            nn.SiLU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            InstanceNorm(256),
            nn.SiLU()
        )
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            MemoryEfficientResBlock(256) for _ in range(9)
        ])
        
        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            InstanceNorm(128),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1), # 256 = 128 + 128 due to skip
            InstanceNorm(64),
            nn.SiLU()
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(64 + 64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, inputs):
        sar_input, terrain_input = inputs
        
        # Initial processing
        x = self.initial(sar_input)
        
        # Save skip connections
        skip1 = x
        
        # Downsample
        x = self.down1(x)
        skip2 = x
        x = self.down2(x)
        
        # Apply middle blocks
        for block in self.middle_blocks:
            x = block([x, terrain_input])
        
        # Upsample with skip connections
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        
        # Generate output
        x = self.output(x)
        
        return x

class TerrainSpatialLayer(nn.Module):
    def __init__(self, height, width):
        super(TerrainSpatialLayer, self).__init__()
        self.height = height
        self.width = width
        self.dense1 = nn.Linear(len(TERRAIN_TYPES), 512)
        self.dense2 = nn.Linear(512, height * width)
        self.relu = nn.ReLU()
    
    def forward(self, terrain_input):
        x = self.relu(self.dense1(terrain_input))
        x = self.dense2(x)
        x = x.view(-1, 1, self.height, self.width)
        return x

# Discriminator model in PyTorch - FIXED VERSION
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Terrain processing
        self.terrain_input_dim = len(TERRAIN_TYPES)
        self.terrain_dense = nn.Linear(self.terrain_input_dim, 512)
        self.terrain_spatial = TerrainSpatialLayer(IMG_HEIGHT, IMG_WIDTH)
        self.relu = nn.ReLU()
        
        # Multi-scale discriminators with correct in/out channels
        self.conv_layers1 = nn.ModuleList()
        self.tan_layers1  = nn.ModuleList()
        self.leaky_layers1= nn.ModuleList()
        self.tga_layers1  = nn.ModuleList()
        self.conv_layers2 = nn.ModuleList()
        self.tan_layers2  = nn.ModuleList()
        self.leaky_layers2= nn.ModuleList()
        self.tga_layers2  = nn.ModuleList()
        
        prev_ch1 = 7   # input_image(3)+target_image(3)+terrain_spatial(1)
        prev_ch2 = 7
        for filters in [64, 128, 256]:
            # Scale 1 (original resolution)
            self.conv_layers1.append(
                spectral_norm(nn.Conv2d(prev_ch1, filters, kernel_size=4, stride=2, padding=1))
            )
            self.tan_layers1.append(TerrainAdaptiveNormalization(filters))
            self.leaky_layers1.append(nn.LeakyReLU(0.2))
            self.tga_layers1.append(TerrainGuidedAttention(filters))
            prev_ch1 = filters

            # Scale 2 (half resolution)
            self.conv_layers2.append(
                spectral_norm(nn.Conv2d(prev_ch2, filters, kernel_size=4, stride=2, padding=1))
            )
            self.tan_layers2.append(TerrainAdaptiveNormalization(filters))
            self.leaky_layers2.append(nn.LeakyReLU(0.2))
            self.tga_layers2.append(TerrainGuidedAttention(filters))
            prev_ch2 = filters
        
        # Output layers (patch discriminator heads)
        self.output1 = spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1))
        self.output2 = spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1))
    
    def forward(self, inputs):
        input_image, target_image, terrain_input = inputs
        
        # Ensure correct shape for terrain_input when using DataParallel
        if len(terrain_input.shape) > 2:
            # Reshape if needed (happens with DataParallel sometimes)
            terrain_input = terrain_input.view(-1, self.terrain_input_dim)
        
        # Process terrain features
        terrain_features = self.relu(self.terrain_dense(terrain_input))
        terrain_spatial = self.terrain_spatial(terrain_input)
        
        # Combine inputs
        x = torch.cat([input_image, target_image, terrain_spatial], dim=1)
        
        # Process at original scale using explicit layer application
        features1 = x
        for i in range(len(self.conv_layers1)):
            # Apply each layer in sequence with proper inputs for terrain-aware layers
            features1 = self.conv_layers1[i](features1)
            features1 = self.tan_layers1[i]([features1, terrain_input])
            features1 = self.leaky_layers1[i](features1)
            features1 = self.tga_layers1[i]([features1, terrain_input])
        
        # Process at half scale
        features2 = F.avg_pool2d(x, 2)
        for i in range(len(self.conv_layers2)):
            # Apply each layer in sequence with proper inputs for terrain-aware layers
            features2 = self.conv_layers2[i](features2)
            features2 = self.tan_layers2[i]([features2, terrain_input])
            features2 = self.leaky_layers2[i](features2)
            features2 = self.tga_layers2[i]([features2, terrain_input])
        
        # Output from both scales
        output1 = self.output1(features1)
        output2 = self.output2(features2)
        
        return [output1, output2]

# Loss functions
def generator_loss(disc_generated_output, generated_images, target_images, generator_model=None):
    """Generator loss function using PyTorch"""
    # Convert to float32 for consistent calculation
    generated_images = generated_images.float()
    target_images = target_images.float()
    
    # GAN loss with BCE
    gan_loss = 0
    for output in disc_generated_output:
        # Clip outputs for stability
        output = torch.clamp(output, min=-20.0, max=20.0)
        # Use ones as labels for generator (wants discriminator to predict real)
        labels = torch.ones_like(output)
        gan_loss += F.binary_cross_entropy_with_logits(output, labels)
    
    gan_loss = gan_loss / len(disc_generated_output)
    
    # L1 loss with clipping
    l1_diff = torch.abs(target_images - generated_images)
    l1_diff = torch.clamp(l1_diff, 0.0, 2.0)
    l1_loss = torch.mean(l1_diff)
    
    # Perceptual loss
    perceptual_loss = compute_perceptual_loss(target_images, generated_images)
    
    # Calculate metrics
    # PSNR
    with torch.no_grad():
        target_clipped = torch.clamp(target_images, -1.0, 1.0)
        generated_clipped = torch.clamp(generated_images, -1.0, 1.0)
        # For PSNR, PyTorch doesn't have a direct equivalent, so we calculate manually
        mse = F.mse_loss(generated_clipped, target_clipped)
        psnr = 10 * torch.log10(4.0 / mse) if mse > 0 else torch.tensor(100.0).to(mse.device)
        
        # For SSIM we'll use PyTorch's implementation (simplified version)
        def gaussian_kernel(size, sigma):
            """Create a gaussian kernel"""
            coords = torch.arange(size).to(device).float()
            coords -= size // 2
            g = coords**2
            g = torch.exp(-(g.reshape(-1, 1) + g.reshape(1, -1)) / (2 * sigma**2))
            g /= g.sum()
            return g.reshape(1, 1, size, size)
        
        def ssim(img1, img2, window_size=11, size_average=True):
            window = gaussian_kernel(window_size, 1.5).repeat(3, 1, 1, 1).to(img1.device)
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=3)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=3)
            mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=3) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=3) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=3) - mu1_mu2
            
            C1, C2 = 0.01**2, 0.03**2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)
        
        ssim_val = ssim(target_clipped, generated_clipped)
    
    # Spectral regularization
    spectral_reg = 0.0
    if generator_model is not None:
        for name, param in generator_model.named_parameters():
            if 'weight' in name:
                # L2 regularization on weight
                spectral_reg += 0.0001 * torch.sum(param ** 2)
    
    # Combine losses
    lambda_val = LAMBDA
    perceptual_weight = 0.05
    total_loss = gan_loss + (lambda_val * l1_loss) + (lambda_val * perceptual_weight * perceptual_loss) + spectral_reg
    
    return (total_loss, gan_loss, l1_loss, perceptual_loss, psnr, ssim_val)

def discriminator_loss(disc_real_output, disc_generated_output):
    """Discriminator loss function for pytorch"""
    # Real loss
    real_loss = 0
    for output in disc_real_output:
        output = torch.clamp(output, -20.0, 20.0)
        # Label smoothing (0.8 instead of 1.0)
        labels = 0.8 * torch.ones_like(output)
        real_loss += F.binary_cross_entropy_with_logits(output, labels)
    
    real_loss = real_loss / len(disc_real_output)
    
    # Fake loss
    fake_loss = 0
    for output in disc_generated_output:
        output = torch.clamp(output, -20.0, 20.0)
        labels = torch.zeros_like(output)
        fake_loss += F.binary_cross_entropy_with_logits(output, labels)
    
    fake_loss = fake_loss / len(disc_generated_output)
    
    # Total loss
    total_loss = real_loss + fake_loss
    
    return total_loss

def compute_gradient_penalty(discriminator, real_images, fake_images, terrain_labels):
    """Calculate gradient penalty for improved WGAN training"""
    # Make sure inputs are float32
    real_images = real_images.float()
    fake_images = fake_images.float()
    terrain_labels = terrain_labels.float()
    
    # Ensure valid range
    real_images = torch.clamp(real_images, -1.0, 1.0)
    fake_images = torch.clamp(fake_images, -1.0, 1.0)
    
    # Get batch size
    batch_size = real_images.size(0)
    
    # Create interpolated images
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_images.device)
    interpolated = real_images + alpha * (fake_images - real_images)
    interpolated.requires_grad_(True)
    
    # Calculate gradients with OOM safeguard
    try:
        disc_interpolates = discriminator([interpolated, interpolated, terrain_labels])
        grad_outputs = [torch.ones_like(output) for output in disc_interpolates]
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device=real_images.device)
        else:
            raise
    
    # Calculate gradient norm
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-10)
    
    # Return gradient penalty
    gp = torch.mean((gradients_norm - 1.0) ** 2) * 0.5
    
    return gp

# Training step function
def train_step(sar_images, color_images, terrain_labels, generator, discriminator,
               generator_optimizer, discriminator_optimizer, scaler):
    """Single training step in PyTorch with mixed precision"""
    # Move to device
    sar_images = sar_images.to(device)
    color_images = color_images.to(device)
    terrain_labels = terrain_labels.to(device)
    
    # Ensure valid range and proper dimensions
    sar_images = torch.clamp(sar_images, -1.0, 1.0)
    color_images = torch.clamp(color_images, -1.0, 1.0)
    
    # Make sure terrain_labels has the correct shape
    if len(terrain_labels.shape) > 2:
        terrain_labels = terrain_labels.view(-1, len(TERRAIN_TYPES))
    
    metrics = {}
    
    try:
        # Generator step
        generator_optimizer.zero_grad()
        
        # Generate images with autocast for mixed precision - updated API
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            generated_images = generator([sar_images, terrain_labels])
            generated_images = torch.clamp(generated_images, -1.0, 1.0)
            
            # Debug print to verify inputs
            if torch.isnan(generated_images).any():
                print("\nDetected NaN in generated images, skipping step")
                return {
                    'gen_total_loss': 0.1,
                    'disc_loss': 0.1
                }, generated_images.detach()
            
            # Discriminator outputs - pass all inputs as a list to match TF implementation
            disc_real_output = discriminator([sar_images, color_images, terrain_labels])
            disc_generated_output = discriminator([sar_images, generated_images, terrain_labels])
            
            # Calculate losses
            gen_loss_result = generator_loss(
                disc_generated_output, generated_images, color_images, generator
            )
            gen_total_loss, gan_loss, l1_loss, perceptual_loss, psnr, ssim = gen_loss_result
            
            # Calculate discriminator loss
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            
            # Gradient penalty
            grad_penalty = compute_gradient_penalty(
                discriminator, color_images, generated_images.detach(), terrain_labels
            )
            
            # Final discriminator loss
            disc_total_loss = disc_loss + 10.0 * grad_penalty
        
        # Scale and backpropagate generator loss
        scaler.scale(gen_total_loss).backward()
        scaler.step(generator_optimizer)
        
        # Discriminator step
        discriminator_optimizer.zero_grad()
        
        # Scale and backpropagate discriminator loss
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # Recalculate discriminator outputs to ensure we have fresh gradients
            disc_real_output = discriminator([sar_images, color_images, terrain_labels])
            disc_generated_output = discriminator([sar_images, generated_images.detach(), terrain_labels])
            
            # Recalculate loss
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            grad_penalty = compute_gradient_penalty(
                discriminator, color_images, generated_images.detach(), terrain_labels
            )
            disc_total_loss = disc_loss + 10.0 * grad_penalty
        
        scaler.scale(disc_total_loss).backward()
        scaler.step(discriminator_optimizer)
        
        # Update scaler
        scaler.update()
        
        # Calculate additional metrics
        with torch.no_grad():
            # Cycle loss: cast to float to avoid half/bias mismatch and catch OOM
            try:
                cycle_reconstructed = generator([generated_images.float(), terrain_labels])
                cycle_loss = torch.mean(torch.abs(sar_images - cycle_reconstructed))
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    cycle_loss = torch.tensor(0.0, device=device)
                else:
                    raise
                    
            # L2 loss
            l2_loss = torch.mean(torch.square(color_images - generated_images))
            
            # Feature matching loss (simplified)
            feature_matching_loss = 0.0
            for real, fake in zip(disc_real_output, disc_generated_output):
                feature_matching_loss += torch.mean(torch.abs(real.detach() - fake))
            feature_matching_loss /= len(disc_real_output)
            
            # LPIPS (approximated)
            lpips = perceptual_loss
        
    except Exception as e:
        print(f"\nException in train step: {e}")
        import traceback
        traceback.print_exc()
        
        # Return defaults
        metrics = {
            'gen_total_loss': 0.1,
            'disc_loss': 0.1,
            'l1_loss': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'cycle_loss': 0.0,
            'l2_loss': 0.0,
            'feature_matching_loss': 0.0,
            'lpips': 0.0,
        }
        try:
            return metrics, generated_images.detach()
        except:
            return metrics, sar_images.detach()
    
    return metrics, generated_images.detach()

# Main training function
def train(train_loader, val_loader, epochs, resume_training=True):
    """Main training loop with PyTorch"""
    # Create visualization directory
    visualization_dir = '/kaggle/working/visualizations'
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Initialize models, history and optimizers
    if resume_training:
        generator, discriminator, history = load_latest_models()
        if generator is not None:
            start_epoch = history.get('epoch', 0) + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            generator = Generator().to(device)
            discriminator = Discriminator().to(device)
            start_epoch = 0
            history = {'epoch': 0, 'history': {k: [] for k in ['gen_loss', 'disc_loss', 'psnr', 'ssim', 'cycle_loss',
                                                              'l2_loss', 'feature_matching_loss', 'lpips', 'l1_loss',
                                                              'val_gen_loss', 'val_disc_loss', 'val_psnr', 'val_ssim',
                                                              'val_l1_loss', 'val_l2_loss', 'val_cycle_loss',
                                                              'val_feature_matching_loss', 'val_lpips', 'fid', 'val_fid']}}
    else:
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        start_epoch = 0
        history = {'epoch': 0, 'history': {k: [] for k in ['gen_loss', 'disc_loss', 'psnr', 'ssim', 'cycle_loss',
                                                          'l2_loss', 'feature_matching_loss', 'lpips', 'l1_loss',
                                                          'val_gen_loss', 'val_disc_loss', 'val_psnr', 'val_ssim',
                                                          'val_l1_loss', 'val_l2_loss', 'val_cycle_loss',
                                                          'val_feature_matching_loss', 'val_lpips', 'fid', 'val_fid']}}
    
    # Setup multi-GPU with proper device handling
    if num_gpus > 1:
        print(f"Setting up DataParallel with {num_gpus} GPUs")
        
        # First make sure all models are on the same device
        for model in [generator, discriminator]:
            # Move all parameters to device:0 first
            for param in model.parameters():
                if param.device != device:
                    param.data = param.data.to(device)
        
        # Now wrap with DataParallel with explicit device_ids
        generator = nn.DataParallel(generator, device_ids=list(range(num_gpus)))
        discriminator = nn.DataParallel(discriminator, device_ids=list(range(num_gpus)))
    
    # Optimizers
    generator_optimizer = optim.Adam(generator.parameters(), lr=1.5e-4, betas=(0.5, 0.999), eps=1e-8)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=8e-5, betas=(0.5, 0.999), eps=1e-8)
    
    # Setup mixed precision training
    scaler = GradScaler()
    
    # Gradient accumulation setup
    grad_accum_steps = GRADIENT_ACCUMULATION_STEPS
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()
        
        # Initialize metrics tracker
        metrics_tracker = MetricsTracker()
        generator.train()
        discriminator.train()
        
        # Process each batch in training dataset
        for step, (sar_batch, color_batch, terrain_batch) in enumerate(train_loader):
            # Skip at zero step to avoid initialization issues
            if step == 0 and epoch == 0:
                continue
            
            # Train step
            batch_metrics, generated_images = train_step(
                sar_batch, color_batch, terrain_batch,
                generator, discriminator,
                generator_optimizer, discriminator_optimizer,
                scaler
            )
            
            # Update metrics
            for k, v in batch_metrics.items():
                metrics_tracker.update(k, v)
            
            # Print progress
            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}, Step {step}, "
                      f"Gen Loss: {metrics_tracker.average('gen_total_loss'):.4f}, "
                      f"Disc Loss: {metrics_tracker.average('disc_loss'):.4f}, "
                      f"PSNR: {metrics_tracker.average('psnr'):.2f}, "
                      f"Time: {elapsed:.2f}s")
        
        # Add metrics to history
        for k, v in metrics_tracker.metrics.items():
            if k in history['history']:
                history['history'][k].append(v / metrics_tracker.counts[k])
        
        # Validation
        print("\nRunning validation...")
        val_metrics_tracker = MetricsTracker()
        generator.eval()
        discriminator.eval()
        
        with torch.no_grad():
            for sar_batch, color_batch, terrain_batch in val_loader:
                # Move to device
                sar_batch = sar_batch.to(device)
                color_batch = color_batch.to(device)
                terrain_batch = terrain_batch.to(device)
                
                # Generate images
                generated_images = generator([sar_batch, terrain_batch])
                
                # Calculate metrics
                disc_real_output = discriminator([sar_batch, color_batch, terrain_batch])
                disc_generated_output = discriminator([sar_batch, generated_images, terrain_batch])
                
                gen_loss_result = generator_loss(
                    disc_generated_output, generated_images, color_batch, generator
                )
                gen_total_loss, _, l1_loss, perceptual_loss, psnr, ssim = gen_loss_result
                
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
                
                # Add to validation metrics
                val_metrics_tracker.update('gen_loss', gen_total_loss.item())
                val_metrics_tracker.update('disc_loss', disc_loss.item())
                val_metrics_tracker.update('psnr', psnr.item())
                val_metrics_tracker.update('ssim', ssim.item())
                val_metrics_tracker.update('l1_loss', l1_loss.item())
                
                # Calculate cycle loss
                cycle_reconstructed = generator([generated_images, terrain_batch])
                cycle_loss = torch.mean(torch.abs(sar_batch - cycle_reconstructed))
                val_metrics_tracker.update('cycle_loss', cycle_loss.item())
                
                # L2 loss
                l2_loss = torch.mean(torch.square(color_batch - generated_images))
                val_metrics_tracker.update('l2_loss', l2_loss.item())
        
        # Add validation metrics to history
        for k, v in val_metrics_tracker.metrics.items():
            val_key = f'val_{k}'
            if val_key in history['history']:
                history['history'][val_key].append(v / val_metrics_tracker.counts[k])
        
        # Print validation metrics
        print(f"Epoch {epoch+1} (Validation): "
              f"Gen Loss: {val_metrics_tracker.average('gen_loss'):.4f}, "
              f"Disc Loss: {val_metrics_tracker.average('disc_loss'):.4f}, "
              f"PSNR: {val_metrics_tracker.average('psnr'):.2f}, "
              f"SSIM: {val_metrics_tracker.average('ssim'):.4f}")
        print(f"Time for epoch {epoch+1}: {time.time() - start_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_model_checkpoint(generator, discriminator, history, epoch)
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Generate visualization
        if (epoch + 1) % 1 == 0:
            save_visualizations(generator, val_loader, epoch, visualization_dir)
        
        # Update history
        history['epoch'] = epoch
    
    return generator, discriminator, history

# Save visualizations function
def save_visualizations(generator, val_loader, epoch, save_dir):
    """Generate and save visualization samples"""
    os.makedirs(save_dir, exist_ok=True)
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch+1}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        # Get a few batches for visualization
        for i, (sar_batch, color_batch, terrain_batch) in enumerate(val_loader):
            if i >= 5:  # Limit to 5 batches
                break
            
            # Move to device
            sar_batch = sar_batch.to(device)
            color_batch = color_batch.to(device)
            terrain_batch = terrain_batch.to(device)
            
            # Generate images
            generated_images = generator([sar_batch, terrain_batch])
            
            # Convert tensors to numpy arrays for visualization
            sar_np = sar_batch.cpu().numpy()
            color_np = color_batch.cpu().numpy()
            generated_np = generated_images.cpu().numpy()
            
            # Create figure
            plt.figure(figsize=(15, 5 * sar_batch.size(0)))
            for j in range(sar_batch.size(0)):
                # Display SAR image
                plt.subplot(sar_batch.size(0), 3, j*3 + 1)
                plt.title(f"Input SAR {j+1}")
                plt.imshow(np.clip((sar_np[j].transpose(1, 2, 0) + 1) / 2, 0, 1))
                plt.axis('off')
                
                # Display ground truth color image
                plt.subplot(sar_batch.size(0), 3, j*3 + 2)
                plt.title(f"Ground Truth {j+1}")
                plt.imshow(np.clip((color_np[j].transpose(1, 2, 0) + 1) / 2, 0, 1))
                plt.axis('off')
                
                # Display generated color image
                plt.subplot(sar_batch.size(0), 3, j*3 + 3)
                plt.title(f"Generated {j+1}")
                plt.imshow(np.clip((generated_np[j].transpose(1, 2, 0) + 1) / 2, 0, 1))
                plt.axis('off')
            
            # Save figure
            plt.savefig(os.path.join(epoch_dir, f'comparison_{i+1}.png'), bbox_inches='tight')
            plt.close()
    
    print(f"Saved visualizations for epoch {epoch+1}")

# Save and load model checkpoints
def save_model_checkpoint(generator, discriminator, history, epoch):
    """Save model checkpoint"""
    checkpoint_dir = '/kaggle/working/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_{timestamp}')
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save model state dicts
    if isinstance(generator, nn.DataParallel):
        torch.save(generator.module.state_dict(), os.path.join(checkpoint_path, 'generator.pt'))
        torch.save(discriminator.module.state_dict(), os.path.join(checkpoint_path, 'discriminator.pt'))
    else:
        torch.save(generator.state_dict(), os.path.join(checkpoint_path, 'generator.pt'))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_path, 'discriminator.pt'))
    
    # Save history
    with open(os.path.join(checkpoint_path, 'history.json'), 'w') as f:
        # Convert any non-serializable values to lists
        serializable_history = history.copy()
        for k, v in history['history'].items():
            serializable_history['history'][k] = list(map(float, v))
        json.dump(serializable_history, f)
    
    # Update latest checkpoint pointer
    with open(os.path.join(checkpoint_dir, 'latest_checkpoint.txt'), 'w') as f:
        f.write(checkpoint_path)
    
    print(f"Models saved to {checkpoint_path}")

def load_latest_models():
    """Load latest model checkpoint"""
    checkpoint_dir = '/kaggle/working/checkpoints'
    latest_file = os.path.join(checkpoint_dir, 'latest_checkpoint.txt')
    
    if not os.path.exists(checkpoint_dir) or not os.path.exists(latest_file):
        print("No checkpoint found. Starting fresh.")
        return None, None, {}
    
    try:
        # Read latest checkpoint path
        with open(latest_file, 'r') as f:
            checkpoint_path = f.read().strip()
        
        # Check if path exists
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint directory {checkpoint_path} not found. Starting fresh.")
            return None, None, {}
        
        # Load model state dicts - initialize on CPU first, then move to correct device
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        generator.load_state_dict(torch.load(os.path.join(checkpoint_path, 'generator.pt')))
        discriminator.load_state_dict(torch.load(os.path.join(checkpoint_path, 'discriminator.pt')))
        
        # Load history
        with open(os.path.join(checkpoint_path, 'history.json'), 'r') as f:
            history = json.load(f)
        
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        return generator, discriminator, history
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None, {}

# Main training execution
if __name__ == '__main__':
    # Create datasets
    train_loader, val_loader, test_loader = create_dataset()
    
    # Train model with better error handling
    try:
        # Print CUDA information
        if torch.cuda.is_available():
            print(f"CUDA Information:")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
        
        # Train model
        generator, discriminator, history = train(train_loader, val_loader, epochs=200, resume_training=True)
        
        # Test evaluation
        generator.eval()
        discriminator.eval()
        test_metrics = MetricsTracker()
        
        with torch.no_grad():
            for sar_batch, color_batch, terrain_batch in test_loader:
                # Move to device
                sar_batch = sar_batch.to(device)
                color_batch = color_batch.to(device)
                terrain_batch = terrain_batch.to(device)
                
                # Generate images
                generated_images = generator([sar_batch, terrain_batch])
                cycle_reconstructed = generator([generated_images, terrain_batch])
                
                # Calculate metrics
                disc_real_output = discriminator([sar_batch, color_batch, terrain_batch])
                disc_generated_output = discriminator([sar_batch, generated_images, terrain_batch])
                
                gen_loss_result = generator_loss(
                    disc_generated_output, generated_images, color_batch, generator
                )
                gen_total_loss, _, l1_loss, perceptual_loss, psnr, ssim = gen_loss_result
                
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
                
                # Calculate cycle loss
                cycle_loss = torch.mean(torch.abs(sar_batch - cycle_reconstructed))
                
                # Calculate L2 loss
                l2_loss = torch.mean(torch.square(color_batch - generated_images))
                
                # Feature matching loss (simplified version similar to TensorFlow code)
                feature_matching_loss = 0.0
                for real_output, fake_output in zip(disc_real_output, disc_generated_output):
                    feature_matching_loss += torch.mean(torch.abs(real_output - fake_output))
                feature_matching_loss /= len(disc_real_output)
                
                # LPIPS (approximated)
                lpips = perceptual_loss
                
                # Update test metrics
                test_metrics.update('gen_total_loss', gen_total_loss.item())
                test_metrics.update('disc_loss', disc_loss.item())
                test_metrics.update('l1_loss', l1_loss.item())
                test_metrics.update('psnr', psnr.item())
                test_metrics.update('ssim', ssim.item())
                test_metrics.update('cycle_loss', cycle_loss.item())
                test_metrics.update('l2_loss', l2_loss.item())
                test_metrics.update('feature_matching_loss', feature_matching_loss.item())
                test_metrics.update('lpips', lpips.item())
        
        print("Test Metrics:")
        for k, v in test_metrics.metrics.items():
            print(f"  {k}: {v / test_metrics.counts[k]:.4f}")
            
    except Exception as e:
        print(f"Error in training/evaluation: {e}")
        import traceback
        traceback.print_exc()
```

