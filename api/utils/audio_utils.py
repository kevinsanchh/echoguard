import torch
import torch.nn.functional as F
import torchaudio
from utils.config import SAMPLE_RATE, MAX_DURATION

# -----------------------
# Audio loading
# -----------------------

def load_audio(path):
    """Load and normalize a WAV file to a fixed duration."""
    wav, sr = torchaudio.load(path)
    
    # Resample to match model's expected sample rate
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # Convert to mono
    wav = wav.mean(dim=0, keepdim=True)
    
    # Pad or trim to fixed length
    max_length = SAMPLE_RATE * MAX_DURATION
    if wav.shape[1] > max_length:
        wav = wav[:, :max_length]
    else:
        wav = F.pad(wav, (0, max_length - wav.shape[1]))
    
    return wav


# -----------------------
# Preprocessing
# -----------------------

def preprocess(wav):
    """Convert waveform into a normalized mel spectrogram."""
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    mel = transform(wav)
    mel_db = torchaudio.functional.amplitude_to_DB(
        mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0
    )

    # Normalize to mean 0, std 1
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # Repeat to create 3-channel input for ResNet
    mel_db = mel_db.repeat(3, 1, 1)

    # Add batch dimension (1, 3, H, W)
    return mel_db.unsqueeze(0)