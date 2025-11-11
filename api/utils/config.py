from pathlib import Path
import torch

# -----------------------
# Global configuration
# -----------------------

# Model + class file paths
MODEL_PATH = Path("assets/audio_resnet_model_best.pth")
CSV_PATH = Path("assets/dataset_train.csv")

# Audio processing parameters
SAMPLE_RATE = 32000
MAX_DURATION = 5  # seconds

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
