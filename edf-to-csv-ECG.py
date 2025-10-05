import os
import numpy as np
import pandas as pd
import mne

# Folder paths
DATA_PATH = "Drowsiness-Signal"
OUTPUT_PATH = "ECG-CSV"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Sampling details
sfreq = 128
expected_samples = sfreq * 2 * 3600  # 921600

# Get all EDF files (excluding annotation files)
edf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".edf") and "annotations" not in f]

for edf_file in sorted(edf_files):
    base_name = edf_file.replace(".edf", "")
    print(f"\nProcessing: {edf_file}")

    # Corresponding annotation file
    ann_file = os.path.join(DATA_PATH, base_name + "_annotations.edf")
    edf_path = os.path.join(DATA_PATH, edf_file)

    # Load signal
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    channels = raw.ch_names

    # Get ECG channel (usually named EKG)
    if "EKG" not in channels:
        print(f"⚠️ EKG channel not found in {edf_file}. Skipping...")
        continue

    ekg_idx = channels.index("EKG")
    signal = raw.get_data(picks=[ekg_idx]).flatten()

    # Handle length mismatches
    actual_len = len(signal)
    if actual_len > expected_samples:
        signal = signal[:expected_samples]
    elif actual_len < expected_samples:
        diff = expected_samples - actual_len
        # Pad with random noise around mean/std of the signal
        pad_values = np.random.normal(signal.mean(), signal.std(), diff)
        signal = np.concatenate([signal, pad_values])
    
    # Initialize labels
    labels = np.zeros(expected_samples, dtype=int)

    # Load annotations
    if os.path.exists(ann_file):
        try:
            ann = mne.read_annotations(ann_file)
            for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
                start_idx = int(onset * sfreq)
                end_idx = int((onset + 1) * sfreq)
                if start_idx < expected_samples:
                    labels[start_idx:min(end_idx, expected_samples)] = 1
        except Exception as e:
            print(f"⚠️ Failed to read annotations for {edf_file}: {e}")
    else:
        print(f"⚠️ Annotation file not found for {edf_file}")

    # Create timestamps (0–7199 seconds, each repeated 128 times)
    timestamps = np.repeat(np.arange(0, 2*3600), sfreq)

    # Build DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": signal,
        "label": labels
    })

    # Save CSV
    csv_path = os.path.join(OUTPUT_PATH, base_name + ".csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {csv_path} ({len(df)} rows)")
