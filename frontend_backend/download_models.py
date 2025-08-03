import os
import requests
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "classifier": {
        "url": os.getenv("CLASSIFIER_URL"),
        "path": "models/classifier.pth"
    },
    "gan": {
        "url": os.getenv("GAN_URL"),
        "path": "models/gan_model.pth"
    }
}

def download_models():
    os.makedirs("models", exist_ok=True)
    for name, info in MODELS.items():
        if not os.path.exists(info["path"]):
            print(f"Downloading {name}...")
            try:
                r = requests.get(info["url"], stream=True)
                r.raise_for_status()
                with open(info["path"], 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"{name} downloaded to {info['path']}")
            except Exception as e:
                print(f"Failed to download {name}: {e}")
        else:
            print(f"{name} already exists.")
