import torch
import platform
import sys
import numpy as np
import PIL
from torchvision import __version__ as torchvision_version
import tqdm
import skimage
import matplotlib
import glob
import os
import scipy

def print_versions():
    print("======== Core ML Libraries ========")
    # PyTorch ecosystem
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision_version}")
    
    # CUDA information
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available: No")
    
    print("\n======== Scientific Libraries ========")
    print(f"NumPy: {np.__version__}")
    print(f"PIL (Pillow): {PIL.__version__}")
    print(f"scikit-image: {skimage.__version__}")
    print(f"scipy: {scipy.__version__}")
    
    print("\n======== Utility Libraries ========")
    print(f"tqdm: {tqdm.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
   
    
    print("\n======== System Information ========")
    print(f"Python: {platform.python_version()}")
    print(f"OS: {platform.platform()}")
    print(f"Python path: {sys.executable}")

if __name__ == "__main__":
    print_versions()
    
    # Also print relevant environment variables
    print("\n======== Relevant Environment Variables ========")
    cuda_envs = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']
    for env in cuda_envs:
        if env in os.environ:
            print(f"{env}: {os.environ[env]}")
