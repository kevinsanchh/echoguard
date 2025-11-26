"""
validation_utils.py
"""

import torch


def validate_nonspeech_waveform(waveform: torch.Tensor, sample_rate: int):
    """
    Validate stitched NON-SPEECH waveform.

    Args:
        waveform: Tensor (channels, samples)
        sample_rate: int Hz
    """


    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    num_channels, num_samples = waveform.shape

    if sample_rate <= 0:
        duration_sec = 0.0
    else:
        duration_sec = float(num_samples) / float(sample_rate)

    if num_samples > 0:
        with torch.no_grad():
            max_abs = waveform.abs().max().item()
            rms = torch.sqrt(torch.mean(waveform ** 2)).item()

            # Fraction of samples above 0.05 (tuned for normalized values)
            fraction_loud = torch.mean((waveform.abs() > 0.05).float()).item()
    else:
        max_abs = 0.0
        rms = 0.0
        fraction_loud = 0.0

    stats = {
        "num_channels": num_channels,
        "num_samples": num_samples,
        "duration_sec": duration_sec,
        "max_abs": max_abs,
        "rms": rms,
        "fraction_loud": fraction_loud,
    }

    failure_reasons = []
    is_valid = True

    # 1. Empty waveform
    if num_samples == 0:
        is_valid = False
        failure_reasons.append("empty_waveform")

    # 2. Duration must be ≥ 0.25 sec
    if duration_sec < 0.25:
        is_valid = False
        failure_reasons.append("too_short")

    # 3. RMS ≥ 0.02
    # Rejects muted mic and quiet noise
    if rms < 0.0013:
        is_valid = False
        failure_reasons.append("low_rms")

    # 4. Peak amplitude ≥ 0.05
    # Prevents quiet static or speech remnants boosted by normalization
    if max_abs < 0.015:
        is_valid = False
        failure_reasons.append("low_peak_amplitude")

    # 5. Fraction loud ≥ 0.06
    # Key separation: real non-speech events (gunshots/explosions) exceed this,
    # while speech fragments typically fall below it.
    if fraction_loud < 0.0001:
        is_valid = False
        failure_reasons.append("low_fraction_loud")

    return is_valid, failure_reasons, stats
