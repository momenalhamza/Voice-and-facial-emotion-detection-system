"""Shared audio feature extraction.

Produces a (channels, T) matrix combining:
  - MFCC          (n_mfcc)
  - log-mel       (n_mels)
  - ZCR           (1)
  - log-F0        (1)  — pitch contour, log-scaled (parselmouth)
  - voicing prob  (1)  — strength of pitch candidate (parselmouth)
  - intensity     (1)  — frame energy in dB (parselmouth)
  - jitter local  (1)  — broadcast scalar (parselmouth)
  - shimmer local (1)  — broadcast scalar (parselmouth)
  - HNR mean      (1)  — broadcast scalar (parselmouth)

Total channels = n_mfcc + n_mels + 1 (ZCR) + 6 (prosodic) = 111 by default.

If parselmouth fails for any clip (very short, all-silence, etc.) the prosodic
channels are zero-filled so the pipeline never crashes.
"""
from __future__ import annotations

import warnings

import librosa
import numpy as np

try:
    import parselmouth
    _HAS_PARSELMOUTH = True
except ImportError:  # pragma: no cover
    _HAS_PARSELMOUTH = False


PROSODIC_CHANNELS = 6  # log-F0, voicing, intensity, jitter, shimmer, HNR


def _frame_times(T: int, sr: int, hop: int) -> np.ndarray:
    return np.arange(T) * hop / sr


def _safe_pitch_intensity(y: np.ndarray, sr: int, frame_times: np.ndarray):
    """Return (log_f0, voicing, intensity_db) each shape (T,) — zero-filled on failure."""
    T = frame_times.size
    if not _HAS_PARSELMOUTH:
        return np.zeros(T), np.zeros(T), np.zeros(T)

    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    except Exception:  # noqa: BLE001
        return np.zeros(T), np.zeros(T), np.zeros(T)

    # Pitch: praat 'ac' is robust, 75-500Hz covers adult speech
    log_f0 = np.zeros(T)
    voicing = np.zeros(T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pitch = snd.to_pitch_ac(pitch_floor=75.0, pitch_ceiling=500.0)
            p_times = pitch.xs()
            p_freq = pitch.selected_array["frequency"]
            p_strength = pitch.selected_array["strength"]
            # Replace unvoiced (freq == 0) with NaN, log, then interpolate
            freq = np.where(p_freq > 0, p_freq, np.nan)
            with np.errstate(invalid="ignore"):
                logf = np.where(np.isnan(freq), np.nan, np.log(freq))
            # Interpolate over NaNs (only when at least one voiced frame)
            voiced_mask = ~np.isnan(logf)
            if voiced_mask.any():
                logf[~voiced_mask] = np.interp(
                    np.flatnonzero(~voiced_mask), np.flatnonzero(voiced_mask), logf[voiced_mask]
                )
            else:
                logf = np.zeros_like(logf)
            log_f0 = np.interp(frame_times, p_times, logf, left=0.0, right=0.0)
            voicing = np.interp(frame_times, p_times, p_strength, left=0.0, right=0.0)
        except Exception:  # noqa: BLE001
            pass

    # Intensity in dB
    intensity_db = np.zeros(T)
    try:
        intensity = snd.to_intensity()
        i_times = intensity.xs()
        i_vals = intensity.values.T.squeeze()
        if i_vals.ndim == 0:
            i_vals = np.array([float(i_vals)])
        intensity_db = np.interp(frame_times, i_times, i_vals, left=0.0, right=0.0)
    except Exception:  # noqa: BLE001
        pass

    return log_f0, voicing, intensity_db


def _safe_jitter_shimmer_hnr(y: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Single-utterance summary stats. Returns zeros on failure."""
    if not _HAS_PARSELMOUTH:
        return 0.0, 0.0, 0.0
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0

    jitter, shimmer, hnr_mean = 0.0, 0.0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = float(parselmouth.praat.call(
                point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            ))
            shimmer = float(parselmouth.praat.call(
                [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ))
        except Exception:  # noqa: BLE001
            jitter, shimmer = 0.0, 0.0
        try:
            harmonicity = snd.to_harmonicity_cc()
            vals = harmonicity.values[harmonicity.values > -200]  # praat sentinel
            hnr_mean = float(vals.mean()) if vals.size else 0.0
        except Exception:  # noqa: BLE001
            hnr_mean = 0.0

    # NaNs → 0 (silent or unvoiced utterances)
    return (float(np.nan_to_num(jitter)),
            float(np.nan_to_num(shimmer)),
            float(np.nan_to_num(hnr_mean)))


def extract_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    include_prosody: bool = True,
) -> np.ndarray:
    """Compute the full (channels, T) feature matrix used by AudioNet.

    The MFCC / mel / ZCR layout is preserved bit-for-bit; prosodic channels
    are appended at the bottom. Each feature stream is z-scored independently
    across time (channel-wise z-score).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)

    T = min(mfcc.shape[1], log_mel.shape[1], zcr.shape[1])
    base = np.concatenate([mfcc[:, :T], log_mel[:, :T], zcr[:, :T]], axis=0)

    # Z-score the per-frame block channel-wise (preserves the original behavior).
    mean = base.mean(axis=1, keepdims=True)
    std = base.std(axis=1, keepdims=True) + 1e-6
    base_z = (base - mean) / std

    if not include_prosody:
        return base_z.astype(np.float32)

    times = _frame_times(T, sr, hop_length)
    log_f0, voicing, intensity_db = _safe_pitch_intensity(y, sr, times)
    jitter, shimmer, hnr_mean = _safe_jitter_shimmer_hnr(y, sr)

    # Z-score the three time-varying prosodic streams per-sample.
    timewise = np.stack([log_f0, voicing, intensity_db], axis=0)
    tw_mean = timewise.mean(axis=1, keepdims=True)
    tw_std = timewise.std(axis=1, keepdims=True) + 1e-6
    timewise_z = (timewise - tw_mean) / tw_std

    # Broadcast scalars use fixed scaling — z-score would collapse them to 0
    # because they are constant across time within a single utterance.
    # Typical ranges: jitter local ~0–0.05, shimmer local ~0–0.15, HNR ~0–30 dB.
    broadcast = np.stack([
        np.full(T, jitter * 20.0, dtype=np.float32),    # ~0–1
        np.full(T, shimmer * 10.0, dtype=np.float32),   # ~0–1.5
        np.full(T, hnr_mean / 10.0, dtype=np.float32),  # ~0–3
    ], axis=0)

    feats = np.concatenate([base_z, timewise_z.astype(np.float32), broadcast], axis=0)
    return feats.astype(np.float32)


def expected_channels(n_mfcc: int, n_mels: int, include_prosody: bool = True) -> int:
    base = n_mfcc + n_mels + 1  # +ZCR
    return base + (PROSODIC_CHANNELS if include_prosody else 0)
