import torch
import torch.nn.functional as F
import torchaudio
from utils.config import SAMPLE_RATE, MAX_DURATION


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


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Ensure waveform is [1, N] mono.
    - If 1D: unsqueeze to [1, N]
    - If multi-channel: average across channels.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.to(torch.float32)


def resample_to_16k(waveform: torch.Tensor, orig_sr: int, target_sr: int = 16000):
    """
    Convert any waveform to mono, float32, and target_sr (default 16 kHz).

    Returns:
        resampled_waveform (torch.Tensor [1, N]), new_sample_rate (int)
    """
    waveform = ensure_mono(waveform)

    if orig_sr == target_sr:
        return waveform, target_sr

    resampled = torchaudio.functional.resample(
        waveform,
        orig_sr,
        target_sr
    )
    return resampled, target_sr