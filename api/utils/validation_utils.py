# utils/validation_utils.py

"""
validation_utils.py

Contains audio validation logic for stitched NON-SPEECH segments.
"""

import torch


def validate_nonspeech_waveform(waveform: torch.Tensor, sample_rate: int):
    """
    Validate stitched NON-SPEECH waveform to ensure it contains
    meaningful audio rather than silence or empty data.

    Parameters
    ----------
    waveform : torch.Tensor
        Shape: (channels, samples)
    sample_rate : int
        Sampling rate of the audio (e.g., 16000)

    Returns
    -------
    is_valid : bool
        True if waveform contains meaningful non-speech audio.
    failure_reasons : list[str]
        Reasons why waveform was rejected (if invalid).
    stats : dict
        Dictionary of computed waveform statistics:
            - num_channels
            - num_samples
            - duration_sec
            - max_abs
            - rms
    """

    # Ensure correct shape
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    num_channels, num_samples = waveform.shape

    # Default stats
    if sample_rate <= 0:
        duration_sec = 0.0
    else:
        duration_sec = float(num_samples) / float(sample_rate)

    if num_samples > 0:
        with torch.no_grad():
            max_abs = waveform.abs().max().item()
            rms = torch.sqrt(torch.mean(waveform ** 2)).item()
    else:
        max_abs = 0.0
        rms = 0.0

    stats = {
        "num_channels": num_channels,
        "num_samples": num_samples,
        "duration_sec": duration_sec,
        "max_abs": max_abs,
        "rms": rms,
    }

    # ------------------------------------------------------------
    # Validation thresholds (LENIENT)
    # ------------------------------------------------------------
    failure_reasons = []
    is_valid = True

    # Empty waveform
    if num_samples == 0:
        is_valid = False
        failure_reasons.append("empty_waveform")

    # Duration too short (< 0.25 sec)
    if duration_sec < 0.25:
        is_valid = False
        failure_reasons.append("too_short")

    # RMS energy too low (< 0.0015)
    if rms < 0.0015:
        is_valid = False
        failure_reasons.append("low_rms")

    # Peak amplitude too low (< 0.01)
    if max_abs < 0.01:
        is_valid = False
        failure_reasons.append("low_peak_amplitude")

    return is_valid, failure_reasons, stats