import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, butter, filtfilt, welch

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "PPG-CSV"
OUTPUT_PATH = "PPG-feature-data-5s"
os.makedirs(OUTPUT_PATH, exist_ok=True)

SFREQ = 128
WINDOW_SEC = 5
WINDOW_SIZE = SFREQ * WINDOW_SEC  # 640 samples per segment

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def bandpass_filter(signal, lowcut=0.5, highcut=8.0, fs=128, order=4):
    """
    Bandpass filter optimized for PPG:
    - Lower bound: remove baseline wander (<0.5 Hz)
    - Upper bound: remove high-frequency noise (>8 Hz)
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def extract_features_ppg(segment):
    """Compute statistical + PPG-domain features for 5s PPG window."""
    features = {}

    # === Statistical features (10) ===
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

    # === Time-domain features (10) ===
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

    # === Amplitude-domain features (5) ===
    features['mean_amplitude'] = np.mean(np.abs(segment))
    features['peak_to_peak'] = np.max(segment) - np.min(segment)
    features['amplitude_entropy'] = entropy(np.abs(segment) + 1e-10)
    features['positive_ratio'] = np.mean(segment > 0)
    features['negative_ratio'] = np.mean(segment < 0)

    # === PPG-domain specific features (10) ===
    filtered = bandpass_filter(segment)

    # Detect systolic peaks (R-like peaks for PPG)
    peaks, _ = find_peaks(filtered, distance=int(0.4 * SFREQ), height=np.mean(filtered) + 0.3*np.std(filtered))
    troughs, _ = find_peaks(-filtered, distance=int(0.4 * SFREQ))

    beat_count = len(peaks)
    features['peak_count'] = beat_count
    features['heart_rate_bpm'] = beat_count * (60 / WINDOW_SEC)  # beats per minute

    if beat_count > 1:
        ibi = np.diff(peaks) / SFREQ  # seconds
        features['ibi_mean'] = np.mean(ibi)
        features['ibi_std'] = np.std(ibi)
    else:
        features['ibi_mean'] = 0
        features['ibi_std'] = 0

    # Pulse amplitude (difference between peak and nearest trough)
    if beat_count > 0 and len(troughs) > 0:
        paired_troughs = []
        for p in peaks:
            nearby_troughs = troughs[np.abs(troughs - p) < SFREQ * 0.5]
            if len(nearby_troughs) > 0:
                paired_troughs.append(np.min(np.abs(filtered[p] - filtered[nearby_troughs])))
        features['pulse_amplitude_mean'] = np.mean(paired_troughs) if paired_troughs else 0
    else:
        features['pulse_amplitude_mean'] = 0

    # Signal morphology
    features['signal_area'] = np.sum(np.abs(filtered))
    features['signal_power'] = np.sum(filtered ** 2) / len(filtered)

    # Spectral domain
    f, Pxx = welch(filtered, fs=SFREQ, nperseg=min(WINDOW_SIZE, len(filtered)))
    features['dominant_freq'] = f[np.argmax(Pxx)]
    psd_norm = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else np.zeros_like(Pxx)
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    # Rise and fall time (approximation)
    features['rise_time'] = np.mean(np.diff(peaks)) / SFREQ if beat_count > 1 else 0
    features['fall_time'] = features['rise_time']

    return features


# ----------------------------
# MAIN LOOP
# ----------------------------
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

for idx, file in enumerate(sorted(csv_files), start=1):
    file_path = os.path.join(DATA_PATH, file)
    print(f"\n[{idx}/{len(csv_files)}] Processing {file} (PPG 5s windows)...")

    df = pd.read_csv(file_path)
    values = df['value'].values
    labels = df['label'].values

    segments, new_labels = [], []

    for i in range(0, len(values), WINDOW_SIZE):
        segment = values[i:i + WINDOW_SIZE]
        if len(segment) < WINDOW_SIZE:
            break
        label = 1 if np.any(labels[i:i + WINDOW_SIZE] == 1) else 0
        segments.append(segment)
        new_labels.append(label)

    # Extract features
    feature_rows = [extract_features_ppg(seg) for seg in segments]
    feature_df = pd.DataFrame(feature_rows)
    feature_df['label'] = new_labels

    # Save
    output_name = file.replace(".csv", "_ppg_feature5s.csv")
    output_path = os.path.join(OUTPUT_PATH, output_name)
    feature_df.to_csv(output_path, index=False)
    print(f"✅ Saved {output_path}  → shape: {feature_df.shape}")
