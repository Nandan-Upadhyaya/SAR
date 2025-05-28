import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.models.inception import inception_v3, Inception_V3_Weights
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
IMG_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 200
LR = 2e-4
LAMBDA_L1 = 100
LAMBDA_PERCEPTUAL = 1
LAMBDA_FM = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')  # Make path absolute

# Print dataset info ONCE
if __name__ == '__main__':
    print(f"Looking for dataset at: {DATASET_ROOT}")
    for terrain in TERRAIN_TYPES:
        sar_dir = os.path.join(DATASET_ROOT, terrain, 'SAR')
        color_dir = os.path.join(DATASET_ROOT, terrain, 'Color')
        sar_files = glob.glob(os.path.join(sar_dir, '*'))
        color_files = glob.glob(os.path.join(color_dir, '*'))
        print(f"{terrain}: {len(sar_files)} SAR files, {len(color_files)} Color files")
        if not os.path.exists(sar_dir):
            print(f"  WARNING: {sar_dir} does not exist")
        if not os.path.exists(color_dir):
            print(f"  WARNING: {color_dir} does not exist")

# ------------------- DATASET -------------------
class SARColorDataset(Dataset):
    def __init__(self, root, split='train', img_size=IMG_SIZE, terrain_types=TERRAIN_TYPES):
        self.samples = []
        self.terrain_types = terrain_types
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        all_pairs = []
        
        print("Building dataset...")
        for terrain in terrain_types:
            sar_dir = os.path.join(root, terrain, 'SAR')
            color_dir = os.path.join(root, terrain, 'Color')
            
            # Check if directories exist
            if not os.path.exists(sar_dir) or not os.path.exists(color_dir):
                print(f"Skipping {terrain}: directory doesn't exist")
                continue
                
            # Get the file lists
            sar_imgs = sorted(glob.glob(os.path.join(sar_dir, '*')))
            color_imgs = sorted(glob.glob(os.path.join(color_dir, '*')))
            
            if len(sar_imgs) == 0 or len(color_imgs) == 0:
                print(f"No images found in {terrain}")
                continue
            
            # Match files - make sure they have the same number
            # For the same terrain, assuming files are already in corresponding order
            num_files = min(len(sar_imgs), len(color_imgs))
            
            print(f"Found {num_files} matching pairs for {terrain}")
            
            # Add pairs by index rather than filename matching
            for i in range(num_files):
                all_pairs.append({
                    'sar': sar_imgs[i],
                    'color': color_imgs[i],
                    'terrain': terrain_types.index(terrain)
                })
        
        print(f"Total pairs found: {len(all_pairs)}")
        
        # Shuffle before splitting to ensure randomness
        np.random.seed(42)
        np.random.shuffle(all_pairs)
        
        n = len(all_pairs)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        if split == 'train':
            selected = all_pairs[:n_train]
        elif split == 'val':
            selected = all_pairs[n_train:n_train + n_val]
        else:
            selected = all_pairs[n_train + n_val:]
        
        self.samples = selected
        print(f"{split} dataset size: {len(self.samples)}")
        
        if len(self.samples) == 0:
            print("WARNING: Dataset is empty! Check your file matching.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sar_img = Image.open(sample['sar']).convert('RGB')
        color_img = Image.open(sample['color']).convert('RGB')
        sar = self.transform(sar_img)
        color = self.transform(color_img)
        terrain = torch.zeros(len(self.terrain_types))
        terrain[sample['terrain']] = 1.0
        return sar, color, terrain

def get_loaders(root, batch_size=BATCH_SIZE):
    train_ds = SARColorDataset(root, 'train')
    val_ds = SARColorDataset(root, 'val')
    test_ds = SARColorDataset(root, 'test')

    # Check for empty datasets and warn the user
    if len(train_ds) == 0:
        raise ValueError(f"Train dataset is empty! Check your DATASET_ROOT ({root}) and directory structure.")
    if len(val_ds) == 0:
        raise ValueError(f"Validation dataset is empty! Check your DATASET_ROOT ({root}) and directory structure.")
    if len(test_ds) == 0:
        print("Warning: Test dataset is empty!")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
    return train_loader, val_loader, test_loader

# ------------------- MODEL -------------------
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
        self.enc1 = UNetBlock(3, 64, down=True, use_bn=False)   # 64x64
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

class Discriminator(nn.Module):
    def __init__(self, terrain_dim=len(TERRAIN_TYPES)):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 1, 1)
        self.terrain_fc = nn.Linear(terrain_dim, 512)
        self.final = nn.Conv2d(512, 1, 4, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, sar, color, terrain):
        x = torch.cat([sar, color], 1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        t = self.terrain_fc(terrain).unsqueeze(2).unsqueeze(3)
        x = x + t
        return self.final(x)

# ------------------- LOSS -------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Fix VGG16 weights warning
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, x, y):
        # x, y: [-1,1] -> [0,1] -> normalize
        x = (x + 1) / 2
        y = (y + 1) / 2
        x = self.transform(x)
        y = self.transform(y)
        return F.l1_loss(self.vgg(x), self.vgg(y))

def gan_loss(pred, target_is_real):
    if target_is_real:
        return F.mse_loss(pred, torch.ones_like(pred))
    else:
        return F.mse_loss(pred, torch.zeros_like(pred))

# Improved IS/FID utilities
def compute_inception_features(images, inception_model, device, return_logits=False):
    # images: (N, 3, H, W) in [0,1]
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        if return_logits:
            pred = inception_model(images.to(device))
            return pred.cpu().numpy()
        else:
            # Use pool3 features for FID
            features = inception_model(images.to(device))
            return features.cpu().numpy()

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def compute_is(preds, splits=10):
    # preds: (N, 1000) logits
    preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    preds = preds / np.sum(preds, axis=1, keepdims=True)
    split_scores = []
    N = preds.shape[0]
    for k in range(splits):
        part = preds[k * N // splits: (k+1) * N // splits, :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * (np.log(pyx + 1e-8) - np.log(py + 1e-8))))
        split_scores.append(np.exp(np.mean(scores)))
    return float(np.mean(split_scores)), float(np.std(split_scores))

def compute_metrics(loader, netG, device, inception_model_logits, inception_model_pool):
    psnr_list, ssim_list = [], []
    real_feats, fake_feats = [], []
    is_logits = []
    for sar, color, terrain in loader:
        sar, color, terrain = sar.to(device), color.to(device), terrain.to(device)
        with torch.no_grad():
            fake = netG(sar, terrain)
        fake_np = ((fake[0].cpu().numpy().transpose(1,2,0) + 1) / 2).clip(0,1)
        color_np = ((color[0].cpu().numpy().transpose(1,2,0) + 1) / 2).clip(0,1)
        psnr = compare_psnr(color_np, fake_np, data_range=1)
        try:
            ssim = compare_ssim(color_np, fake_np, channel_axis=2, data_range=1)
        except TypeError:
            ssim = compare_ssim(color_np, fake_np, multichannel=True, data_range=1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        # IS: use logits, FID: use pool3 features
        real_img = (color + 1) / 2
        fake_img = (fake + 1) / 2
        real_feat = compute_inception_features(real_img, inception_model_pool, device, return_logits=False)
        fake_feat = compute_inception_features(fake_img, inception_model_pool, device, return_logits=False)
        real_feats.append(real_feat)
        fake_feats.append(fake_feat)
        fake_logits = compute_inception_features(fake_img, inception_model_logits, device, return_logits=True)
        is_logits.append(fake_logits)
    real_feats = np.concatenate(real_feats, axis=0)
    fake_feats = np.concatenate(fake_feats, axis=0)
    is_logits = np.concatenate(is_logits, axis=0)
    mu_real, sigma_real = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    is_mean, is_std = compute_is(is_logits)
    return np.mean(psnr_list), np.mean(ssim_list), is_mean, fid

# ------------------- TRAINING -------------------
def train():
    train_loader, val_loader, test_loader = get_loaders(DATASET_ROOT, BATCH_SIZE)
    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)
    perceptual_loss = PerceptualLoss().to(DEVICE)
    optG = Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
    optD = Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler()
    best_psnr = 0

    # Inception model for IS/FID
    inception_model_logits = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False, aux_logits=True).to(DEVICE)
    inception_model_logits.eval()
    for p in inception_model_logits.parameters():
        p.requires_grad = False
    inception_model_pool = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception_model_pool.fc = nn.Identity()
    inception_model_pool.eval()
    inception_model_pool.to(DEVICE)
    for p in inception_model_pool.parameters():
        p.requires_grad = False

    # Create checkpoints and samples directories if not exist
    checkpoints_dir = 'checkpoints'
    samples_dir = 'samples'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        netG.train()
        netD.train()
        running_loss_G = 0.0
        running_loss_D = 0.0
        for sar, color, terrain in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            sar, color, terrain = sar.to(DEVICE), color.to(DEVICE), terrain.to(DEVICE)
            # Train D
            with autocast():
                fake = netG(sar, terrain)
                pred_real = netD(sar, color, terrain)
                pred_fake = netD(sar, fake.detach(), terrain)
                loss_D = (gan_loss(pred_real, True) + gan_loss(pred_fake, False)) * 0.5
            optD.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(optD)
            # Train G
            with autocast():
                fake = netG(sar, terrain)
                pred_fake = netD(sar, fake, terrain)
                loss_GAN = gan_loss(pred_fake, True)
                loss_L1 = F.l1_loss(fake, color)
                loss_perc = perceptual_loss(fake, color)
                loss_G = loss_GAN + LAMBDA_L1 * loss_L1 + LAMBDA_PERCEPTUAL * loss_perc
            optG.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(optG)
            scaler.update()
            # Accumulate losses for printing
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
        avg_loss_G = running_loss_G / len(train_loader)
        avg_loss_D = running_loss_D / len(train_loader)
        print(f"Epoch {epoch+1}: Generator Loss: {avg_loss_G:.4f} | Discriminator Loss: {avg_loss_D:.4f}")

        # Compute metrics for train set
        netG.eval()
        print("Computing train metrics...")
        train_psnr, train_ssim, train_is, train_fid = compute_metrics(
            train_loader, netG, DEVICE, inception_model_logits, inception_model_pool)
        print(f"Train PSNR: {train_psnr:.2f} | SSIM: {train_ssim:.4f} | IS: {train_is:.2f} | FID: {train_fid:.2f}")

        # Compute metrics for val set
        print("Computing val metrics...")
        val_psnr, val_ssim, val_is, val_fid = compute_metrics(
            val_loader, netG, DEVICE, inception_model_logits, inception_model_pool)
        print(f"Val PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | IS: {val_is:.2f} | FID: {val_fid:.2f}")

        # Save a sample SAR, GT, and generated image as a single jpg after every epoch
        with torch.no_grad():
            sar, color, terrain = next(iter(val_loader))
            sar, color, terrain = sar.to(DEVICE), color.to(DEVICE), terrain.to(DEVICE)
            fake = netG(sar, terrain)
            # Take first sample in batch
            sar_img = (sar[0] + 1) / 2
            color_img = (color[0] + 1) / 2
            fake_img = (fake[0] + 1) / 2
            # Stack horizontally for visualization
            grid = torch.stack([sar_img, color_img, fake_img], dim=0)
            grid = make_grid(grid, nrow=3)
            np_grid = (grid.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            plt.imsave(os.path.join(samples_dir, f'epoch_{epoch+1}.jpg'), np_grid)

        # Save model checkpoint after every epoch
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optG.state_dict(),
            'optimizerD_state_dict': optD.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_psnr': best_psnr
        }, os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Validation PSNR for best model saving
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(netG.state_dict(), 'best_generator.pth')
            torch.save(netD.state_dict(), 'best_discriminator.pth')
            print("Saved best model.")

    print("Training complete.")

# ------------------- TESTING -------------------
def test():
    _, _, test_loader = get_loaders(DATASET_ROOT, 1)
    netG = Generator().to(DEVICE)
    netG.load_state_dict(torch.load('best_generator.pth', map_location=DEVICE))
    netG.eval()
    os.makedirs('test_results', exist_ok=True)
    with torch.no_grad():
        for i, (sar, color, terrain) in enumerate(test_loader):
            sar, color, terrain = sar.to(DEVICE), color.to(DEVICE), terrain.to(DEVICE)
            fake = netG(sar, terrain)
            save_image((fake+1)/2, f'test_results/gen_{i:04d}.png')
            save_image((color+1)/2, f'test_results/gt_{i:04d}.png')
            save_image((sar+1)/2, f'test_results/sar_{i:04d}.png')
    print("Test images saved to test_results/")

if __name__ == '__main__':
    train()
    test()
