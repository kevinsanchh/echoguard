import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from utils.config import MODEL_PATH, CSV_PATH, DEVICE

# -----------------------
# Model architecture
# -----------------------

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_ch)
        self.relu = torch.nn.ReLU(inplace=True)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetAudio(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = ResidualBlock(3, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        self.layer4 = ResidualBlock(128, 128)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.15)
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# -----------------------
# Model + class loader
# -----------------------

def load_ml_assets():
    """Load model weights and class names once at startup."""
    # Load CSV → get class list
    df = pd.read_csv(CSV_PATH)
    classes = sorted(df["category"].unique())

    # Load trained model weights
    model = ResNetAudio(num_classes=len(classes)).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"DEBUG: ML model loaded successfully on {DEVICE}. Classes: {len(classes)}")
    return model, classes