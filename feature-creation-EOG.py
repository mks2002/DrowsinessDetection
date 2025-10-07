import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, butter, filtfilt, welch


def extract_features_eog(segment):
    """Extract statistical, time, amplitude, and EOG-specific features."""
    features = {}

    # === Statistical (10) ===
    features['mean'] = np.mean(segment)
    features['std'] = np.std(segment)
    features['var'] = np.var(segment)
    features['min'] = np.min(segment)
    features['max'] = np.max(segment)
    features['median'] = np.median(segment)
    features['skew'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    features['energy'] = np.sum(segment ** 2)
    features['rms'] = np.sqrt(np.mean(segment ** 2))

    # === Time-domain (10) ===
    diff = np.diff(segment)
    abs_diff = np.abs(diff)
    features['mean_diff'] = np.mean(diff)
    features['std_diff'] = np.std(diff)
    features['mean_abs_diff'] = np.mean(abs_diff)
    features['max_abs_diff'] = np.max(abs_diff)
    features['zero_crossings'] = ((segment[:-1] * segment[1:]) < 0).sum()
    features['slope_mean'] = np.mean(diff)
    features['slope_std'] = np.std(diff)
    features['signal_range'] = np.ptp(segment)
    features['signal_energy'] = np.sum(segment ** 2)
    features['signal_entropy'] = entropy(np.abs(segment) + 1e-10)

    # === Amplitude-domain (5) ===
    features['mean_amplitude'] = np.mean(np.abs(segment))
    features['peak_to_peak'] = np.max(segment) - np.min(segment)
    features['amplitude_entropy'] = entropy(np.abs(segment) + 1e-10)
    features['positive_ratio'] = np.mean(segment > 0)
    features['negative_ratio'] = np.mean(segment < 0)

    # === EOG-domain features (10) ===
    filtered = bandpass_filter(segment, lowcut=0.1, highcut=15, fs=SFREQ)
    peaks, _ = find_peaks(filtered, distance=SFREQ*0.3, height=np.mean(filtered)+np.std(filtered))
    features['blink_count'] = len(peaks)
    features['blink_rate_per_min'] = len(peaks) * (60 / WINDOW_SEC)
    features['blink_amplitude_mean'] = np.mean(filtered[peaks]) if len(peaks) else 0
    features['blink_amplitude_std'] = np.std(filtered[peaks]) if len(peaks) > 1 else 0
    features['mean_power_0_5Hz'] = bandpower(filtered, SFREQ, (0.1, 5))
    features['mean_power_5_10Hz'] = bandpower(filtered, SFREQ, (5, 10))
    features['low_high_power_ratio'] = (
        features['mean_power_0_5Hz'] / (features['mean_power_5_10Hz'] + 1e-10)
    )
    features['dominant_freq'] = np.argmax(np.abs(np.fft.rfft(filtered))) * (SFREQ / len(filtered))
    features['spectral_entropy'] = entropy(np.abs(filtered) + 1e-10)
    features['area'] = np.sum(np.abs(filtered))

    return features
