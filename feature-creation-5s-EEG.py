import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, butter, filtfilt, welch


def bandpower(data, fs, band):
    from scipy.signal import welch
    low, high = band
    freqs, psd = welch(data, fs=fs, nperseg=fs*2)
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx_band], freqs[idx_band])

def extract_features_eeg(segment):
    """Extract statistical, time, amplitude, and EEG-domain features."""
    features = {}

    # === Basic statistical features (10) ===
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
    features['mean_diff'] = np.mean(diff)
    features['std_diff'] = np.std(diff)
    features['zero_crossings'] = ((segment[:-1] * segment[1:]) < 0).sum()
    features['range'] = np.ptp(segment)
    features['slope_mean'] = np.mean(diff)
    features['slope_std'] = np.std(diff)
    features['mean_abs_diff'] = np.mean(np.abs(diff))
    features['max_abs_diff'] = np.max(np.abs(diff))
    features['signal_energy'] = np.sum(segment ** 2)
    features['signal_entropy'] = entropy(np.abs(segment) + 1e-10)

    # === Amplitude-domain (5) ===
    features['mean_amplitude'] = np.mean(np.abs(segment))
    features['peak_to_peak'] = np.max(segment) - np.min(segment)
    features['positive_ratio'] = np.mean(segment > 0)
    features['negative_ratio'] = np.mean(segment < 0)
    features['amplitude_entropy'] = entropy(np.abs(segment) + 1e-10)

    # === EEG-domain features (10) ===
    # Compute band powers
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    total_power = 0
    for band_name, band_range in bands.items():
        bp = bandpower(segment, SFREQ, band_range)
        features[f'{band_name}_power'] = bp
        total_power += bp

    # Relative bandpowers
    for band_name in bands.keys():
        features[f'{band_name}_rel_power'] = (
            features[f'{band_name}_power'] / total_power if total_power > 0 else 0
        )

    # Ratios indicating drowsiness
    features['theta_alpha_ratio'] = (
        features['theta_power'] / (features['alpha_power'] + 1e-10)
    )
    features['theta_beta_ratio'] = (
        features['theta_power'] / (features['beta_power'] + 1e-10)
    )

    # Hjorth parameters
    diff1 = np.diff(segment)
    diff2 = np.diff(diff1)
    var0, var1, var2 = np.var(segment), np.var(diff1), np.var(diff2)
    features['hjorth_activity'] = var0
    features['hjorth_mobility'] = np.sqrt(var1 / var0) if var0 != 0 else 0
    features['hjorth_complexity'] = (
        np.sqrt(var2 / var1) / features['hjorth_mobility'] if var1 != 0 else 0
    )

    return features
