import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from utils.config import MODEL_PATH, DEVICE, SAMPLE_RATE, MAX_SAMPLES, MAX_DURATION
import os
import argparse
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf 
from pathlib import Path



# -----------------------------------------------------------
# Model architecture (ImprovedResNetAudio)
# -----------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.se_block = SEBlock(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se_block(out)
        out += self.shortcut(identity)
        out = self.silu(out)
        return out


class ImprovedResNetAudio(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        # Initial conv stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2, dropout=dropout)

        # Dual pooling + classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),  # multi-label logits
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = []
        layers.append(
            ResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                dropout=dropout,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    dropout=dropout,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 1, mel_bins, time)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_out = self.avgpool(x).flatten(1)
        max_out = self.maxpool(x).flatten(1)
        x = torch.cat((avg_out, max_out), dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        return x


# -----------------------------------------------------------
# Preprocessing helper
# -----------------------------------------------------------

def preprocess_waveform(waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
    """
    Convert a raw waveform tensor into a normalized mel-spectrogram batch
    suitable for ImprovedResNetAudio.

    Args:
        waveform: Tensor of shape (samples,) or (channels, samples)
        orig_sample_rate: Original sampling rate of the waveform

    Returns:
        mel_batch: Tensor of shape (1, 1, n_mels, time)
    """
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)

    waveform = waveform.float()

    # Ensure shape is (channels, samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)

    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16k if needed
    if orig_sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=orig_sample_rate,
            new_freq=SAMPLE_RATE,
        )

    # Enforce fixed duration (pad or truncate)
    num_samples = waveform.shape[1]
    if num_samples > MAX_SAMPLES:
        waveform = waveform[:, :MAX_SAMPLES]
    else:
        pad_amount = MAX_SAMPLES - num_samples
        waveform = F.pad(waveform, (0, pad_amount))

    # MelSpectrogram (same config as teammate's preprocess_audio)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=256,
        n_mels=128,
        f_min=20,
        f_max=8000,
        window_fn=torch.hann_window,
        power=2.0,
        norm="slaney",
        mel_scale="htk",
    )

    mel = transform(waveform)  # (1, n_mels, time)

    # Convert to dB
    mel_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)

    # Normalize (mean/std)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # Add batch dimension for model: (1, 1, n_mels, time)
    mel_batch = mel_norm.unsqueeze(0)
    return mel_batch


# -----------------------------------------------------------
# Model loader
# -----------------------------------------------------------

def load_model(model_path: str | Path = None):
    """
    Load the improved multi-label model and its class names from a checkpoint.

    Args:
        model_path: Optional explicit path. If None, uses MODEL_PATH from config.

    Returns:
        model: ImprovedResNetAudio on DEVICE, in eval mode
        class_names: list[str]
    """
    if model_path is None:
        model_path = MODEL_PATH

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"[ModelLoader] Model file not found at {model_path}")

    print(f"[ModelLoader] Loading improved model from {model_path} on {DEVICE}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if "classes" not in checkpoint:
        raise KeyError(
            "[ModelLoader] Checkpoint is missing 'classes' list. "
            "The new multi-label model is expected to store class names "
            "under the 'classes' key."
        )

    class_names = checkpoint["classes"]

    model = ImprovedResNetAudio(num_classes=len(class_names)).to(DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"[ModelLoader] ML model loaded successfully on {DEVICE}. "
        f"Num classes={len(class_names)}"
    )

    return model, class_names


# -----------------------------------------------------------
# Prediction helper
# -----------------------------------------------------------

def predict(
    model: torch.nn.Module,
    class_names: list[str],
    mel_batch: torch.Tensor,
    threshold: float = 0.70,
):
    """
    Run multi-label prediction on a mel-spectrogram batch.

    Args:
        model: ImprovedResNetAudio
        class_names: list of class names
        mel_batch: Tensor of shape (1, 1, n_mels, time)
        threshold: probability threshold in [0,1] for including a class in detections.

    Returns:
        result: dict with:
            - top_class: str
            - top_score: float (0–1)
            - detections: list[(class_name: str, score: float)] sorted by score desc
    """
    model.eval()
    mel_batch = mel_batch.to(DEVICE)

    with torch.no_grad():
        logits = model(mel_batch)          # (1, num_classes)
        probs = torch.sigmoid(logits).squeeze(0)  # (num_classes,)

    # Top class (for legacy 'prediction' + 'confidence' fields)
    top_idx = int(torch.argmax(probs).item())
    top_score = float(probs[top_idx].item())
    top_class = class_names[top_idx]

    # Thresholded detections (multi-label)
    detections = []
    for idx, score in enumerate(probs):
        score_val = float(score.item())
        if score_val >= threshold:
            detections.append((class_names[idx], score_val))

    detections.sort(key=lambda x: x[1], reverse=True)

    return {
        "top_class": top_class,
        "top_score": top_score,      # 0–1; convert to % in CNN_model
        "detections": detections,    # list of (label, 0–1)
    }