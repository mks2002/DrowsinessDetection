import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import butter, filtfilt, welch, find_peaks

# ===============================================================
# CONFIGURATION
# ===============================================================
EEG_PATH = "EEG-CSV"
EOG_PATH = "EOG-CSV"
EEG_OUT = "feature-data-EEG"
EOG_OUT = "feature-data-EOG"

os.makedirs(EEG_OUT, exist_ok=True)
os.makedirs(EOG_OUT, exist_ok=True)

SFREQ = 128             # samples per second
WINDOW_SEC = 5          # seconds per segment
WINDOW_SIZE = SFREQ * WINDOW_SEC  # 640 samples

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================


def bandpass_filter(signal, lowcut, highcut, fs=SFREQ, order=4):
    """Simple bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)


def bandpower(data, fs, band):
    """Power spectral density within specific frequency band."""
    low, high = band
    freqs, psd = welch(data, fs=fs, nperseg=min(fs*2, len(data)))
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx_band], freqs[idx_band])

# ===============================================================
# FEATURE EXTRACTORS
# ===============================================================


def extract_features_eeg(segment):
    """Extract EEG features: statistical + time + amplitude + bandpower + Hjorth."""
    f = {}

    # ---- Statistical features ----
    f['mean'] = np.mean(segment)
    f['std'] = np.std(segment)
    f['var'] = np.var(segment)
    f['min'] = np.min(segment)
    f['max'] = np.max(segment)
    f['median'] = np.median(segment)
    f['skew'] = skew(segment)
    f['kurtosis'] = kurtosis(segment)
    f['energy'] = np.sum(segment**2)
    f['rms'] = np.sqrt(np.mean(segment**2))

    # ---- Time-domain features ----
    diff = np.diff(segment)
    abs_diff = np.abs(diff)
    f['mean_diff'] = np.mean(diff)
    f['std_diff'] = np.std(diff)
    f['mean_abs_diff'] = np.mean(abs_diff)
    f['max_abs_diff'] = np.max(abs_diff)
    f['zero_cross'] = ((segment[:-1]*segment[1:]) < 0).sum()
    f['range'] = np.ptp(segment)
    f['slope_mean'] = np.mean(diff)
    f['slope_std'] = np.std(diff)
    f['signal_entropy'] = entropy(np.abs(segment)+1e-10)
    f['signal_energy'] = np.sum(segment**2)

    # ---- Amplitude features ----
    f['mean_amplitude'] = np.mean(np.abs(segment))
    f['peak_to_peak'] = np.max(segment)-np.min(segment)
    f['positive_ratio'] = np.mean(segment > 0)
    f['negative_ratio'] = np.mean(segment < 0)
    f['amplitude_entropy'] = entropy(np.abs(segment)+1e-10)

    # ---- EEG domain features ----
    # Frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    total_power = 0
    for bname, brange in bands.items():
        bp = bandpower(segment, SFREQ, brange)
        f[f'{bname}_power'] = bp
        total_power += bp

    for bname in bands.keys():
        f[f'{bname}_rel'] = f[f'{bname}_power'] / \
            total_power if total_power > 0 else 0

    # Ratios for drowsiness detection
    f['theta_alpha_ratio'] = f['theta_power']/(f['alpha_power']+1e-10)
    f['theta_beta_ratio'] = f['theta_power']/(f['beta_power']+1e-10)

    # Hjorth parameters
    diff1 = np.diff(segment)
    diff2 = np.diff(diff1)
    var0, var1, var2 = np.var(segment), np.var(diff1), np.var(diff2)
    f['hjorth_activity'] = var0
    f['hjorth_mobility'] = np.sqrt(var1/var0) if var0 != 0 else 0
    f['hjorth_complexity'] = (np.sqrt(
        var2/var1)/f['hjorth_mobility']) if (var1 != 0 and f['hjorth_mobility'] != 0) else 0

    return f


def extract_features_eog(segment):
    """Extract EOG features: blink-based + slow eye movement + stats."""
    f = {}

    # ---- Statistical ----
    f['mean'] = np.mean(segment)
    f['std'] = np.std(segment)
    f['var'] = np.var(segment)
    f['min'] = np.min(segment)
    f['max'] = np.max(segment)
    f['median'] = np.median(segment)
    f['skew'] = skew(segment)
    f['kurtosis'] = kurtosis(segment)
    f['energy'] = np.sum(segment**2)
    f['rms'] = np.sqrt(np.mean(segment**2))

    # ---- Time-domain ----
    diff = np.diff(segment)
    abs_diff = np.abs(diff)
    f['mean_diff'] = np.mean(diff)
    f['std_diff'] = np.std(diff)
    f['mean_abs_diff'] = np.mean(abs_diff)
    f['max_abs_diff'] = np.max(abs_diff)
    f['zero_cross'] = ((segment[:-1]*segment[1:]) < 0).sum()
    f['slope_mean'] = np.mean(diff)
    f['slope_std'] = np.std(diff)
    f['signal_range'] = np.ptp(segment)
    f['signal_entropy'] = entropy(np.abs(segment)+1e-10)
    f['signal_energy'] = np.sum(segment**2)

    # ---- Amplitude ----
    f['mean_amplitude'] = np.mean(np.abs(segment))
    f['peak_to_peak'] = np.max(segment)-np.min(segment)
    f['amplitude_entropy'] = entropy(np.abs(segment)+1e-10)
    f['positive_ratio'] = np.mean(segment > 0)
    f['negative_ratio'] = np.mean(segment < 0)

    # ---- EOG specific ----
    filtered = bandpass_filter(segment, 0.1, 15)
    peaks, _ = find_peaks(filtered, distance=SFREQ*0.3,
                          height=np.mean(filtered)+np.std(filtered))
    f['blink_count'] = len(peaks)
    f['blink_rate_per_min'] = len(peaks)*(60/WINDOW_SEC)
    f['blink_amp_mean'] = np.mean(filtered[peaks]) if len(peaks) > 0 else 0
    f['blink_amp_std'] = np.std(filtered[peaks]) if len(peaks) > 1 else 0

    f['low_freq_power'] = bandpower(filtered, SFREQ, (0.1, 5))
    f['high_freq_power'] = bandpower(filtered, SFREQ, (5, 10))
    f['low_high_ratio'] = f['low_freq_power']/(f['high_freq_power']+1e-10)
    f['dominant_freq'] = np.argmax(
        np.abs(np.fft.rfft(filtered)))*(SFREQ/len(filtered))
    f['spectral_entropy'] = entropy(np.abs(filtered)+1e-10)
    f['area'] = np.sum(np.abs(filtered))
    return f

# ===============================================================
# GENERIC PIPELINE FUNCTION
# ===============================================================


def process_modality(input_path, output_path, n_channels, channel_prefix, extractor):
    """Convert multichannel EEG/EOG CSV into per-window feature DataFrame."""
    csv_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    for idx, file in enumerate(sorted(csv_files), start=1):
        print(f"\n[{idx}/{len(csv_files)}] Processing {file} ...")
        df = pd.read_csv(os.path.join(input_path, file))

        # Expect columns: timestamp + channel1 + channel2 + ... + label
        label = df['label'].values
        channel_cols = [
            c for c in df.columns if c not in ['timestamp', 'label']]
        if len(channel_cols) != n_channels:
            print(
                f"⚠️ Warning: Expected {n_channels} channels but found {len(channel_cols)}.")

        segments, labels = [], []
        for i in range(0, len(df), WINDOW_SIZE):
            if i+WINDOW_SIZE > len(df):
                break
            labels.append(1 if np.any(label[i:i+WINDOW_SIZE] == 1) else 0)
            # combine all channels' features in one row
            feats_allch = {}
            for ch in channel_cols:
                seg = df[ch].values[i:i+WINDOW_SIZE]
                ch_feats = extractor(seg)
                feats_allch.update(
                    {f"{ch}_{k}": v for k, v in ch_feats.items()})
            segments.append(feats_allch)

        feature_df = pd.DataFrame(segments)
        feature_df['label'] = labels
        out_file = file.replace(".csv", f"_feature.csv")
        out_path = os.path.join(output_path, out_file)
        feature_df.to_csv(out_path, index=False)
        print(f"✅ Saved {out_path} → shape: {feature_df.shape}")


# ===============================================================
# EXECUTE PIPELINES
# ===============================================================
print("\n===== Extracting EEG features =====")
process_modality(EEG_PATH, EEG_OUT, n_channels=4,
                 channel_prefix='EEG', extractor=extract_features_eeg)

print("\n===== Extracting EOG features =====")
process_modality(EOG_PATH, EOG_OUT, n_channels=2,
                 channel_prefix='EOG', extractor=extract_features_eog)

print("\n✅ All feature files generated successfully!")
