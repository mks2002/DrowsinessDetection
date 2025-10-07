import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, butter, filtfilt, welch

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "ECG-CSV"
OUTPUT_PATH = "feature-data-5second"
os.makedirs(OUTPUT_PATH, exist_ok=True)

SFREQ = 128                         # samples per second
WINDOW_SEC = 5                      # window length in seconds
WINDOW_SIZE = SFREQ * WINDOW_SEC    # 640 samples per segment
SECONDS = 2 * 3600                  # 7200 seconds per file

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def bandpass_filter(signal, lowcut=5.0, highcut=40.0, fs=128, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_features(segment):
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

    # === Time-domain ECG-like features (10) ===
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

    # === Amplitude domain features (5) ===
    features['mean_amplitude'] = np.mean(np.abs(segment))
    features['peak_to_peak'] = np.max(segment) - np.min(segment)
    features['amplitude_entropy'] = entropy(np.abs(segment) + 1e-10)
    features['positive_ratio'] = np.mean(segment > 0)
    features['negative_ratio'] = np.mean(segment < 0)

    # === ECG-domain physiological features (10) ===
    filtered = bandpass_filter(segment)
    peaks, _ = find_peaks(filtered, distance=20, height=np.mean(filtered) + np.std(filtered))
    beat_count = len(peaks)
    features['beat_count'] = beat_count
    features['avg_heart_rate'] = beat_count * (60 / WINDOW_SEC)  # beats per minute scaled by window length
    features['r_peak_amplitude_mean'] = np.mean(filtered[peaks]) if beat_count > 0 else 0
    features['r_peak_amplitude_std'] = np.std(filtered[peaks]) if beat_count > 1 else 0

    if beat_count > 1:
        rr_intervals = np.diff(peaks)
        features['rr_interval_mean'] = np.mean(rr_intervals)
        features['rr_interval_std'] = np.std(rr_intervals)
    else:
        features['rr_interval_mean'] = 0
        features['rr_interval_std'] = 0

    features['signal_area'] = np.sum(np.abs(segment))
    features['signal_power'] = np.sum(segment ** 2) / len(segment)

    f, Pxx = welch(segment, fs=SFREQ, nperseg=min(WINDOW_SIZE, len(segment)))
    features['dominant_freq'] = f[np.argmax(Pxx)]
    psd_norm = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else np.zeros_like(Pxx)
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    return features

# ----------------------------
# PROCESS EACH CSV FILE
# ----------------------------
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

for idx, file in enumerate(sorted(csv_files), start=1):
    file_path = os.path.join(DATA_PATH, file)
    print(f"\n[{idx}/{len(csv_files)}] Processing {file} (5s windows)...")

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

    # Feature extraction
    feature_rows = [extract_features(seg) for seg in segments]
    feature_df = pd.DataFrame(feature_rows)
    feature_df['label'] = new_labels

    # Save
    output_name = file.replace(".csv", "_feature5s.csv")
    output_path = os.path.join(OUTPUT_PATH, output_name)
    feature_df.to_csv(output_path, index=False)
    print(f"✅ Saved {output_path}  → shape: {feature_df.shape}")
