

import os
import time
import warnings
import numpy as np
import pandas as pd

from mne_features.feature_extraction import extract_features

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLEANED_DIR = "data/cleaned"
OUTPUT_DIR = "data/features"
METADATA_FILE = os.path.join(CLEANED_DIR, "_metadata.csv")

SAMPLING_RATE = 128.0
EPOCH_DURATION_SEC = 5.0
EPOCH_LENGTH_SAMPLES = int(EPOCH_DURATION_SEC * SAMPLING_RATE)  

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]
N_CHANNELS = len(CHANNELS)

# Features to extract via mne-features
# Reference: https://mne.tools/mne-features/api.html
FEATURE_FUNCS = [
    "mean",
    "std",
    "kurtosis",
    "skewness",
    "ptp_amp",          # peak-to-peak amplitude
    "pow_freq_bands",   # band powers (configured below)
    "app_entropy",      # approximate entropy
    "katz_fd",          # Katz fractal dimension
    "hjorth_mobility",
    "hjorth_complexity",
]

# Frequency bands for pow_freq_bands feature
# Delta, theta, alpha, beta, gamma
FREQ_BANDS = np.array([0.5, 4.0, 8.0, 13.0, 30.0, 40.0])

# Feature parameters for mne-features
FEATURE_PARAMS = {
    "pow_freq_bands__freq_bands": FREQ_BANDS,
    "pow_freq_bands__ratios": None,  # compute theta/beta ratio
    "pow_freq_bands__ratios_triu": False,
    "pow_freq_bands__log": False,
    "pow_freq_bands__normalize": True,
    "app_entropy__emb": 2,
}


# ---------------------------------------------------------------------------
# Epoch a single subject's signal
# ---------------------------------------------------------------------------
def segment_subject(signal, epoch_length=EPOCH_LENGTH_SAMPLES):
    
     n_channels, n_samples = signal.shape
     step = epoch_length // 2  # 50% overlap

     starts = list(range(0, n_samples - epoch_length + 1, step))

     if len(starts) == 0:
        return np.empty((0, n_channels, epoch_length))

     epochs = np.array([signal[:, s:s + epoch_length] for s in starts])
     return epochs  # shape: (n_epochs, n_channels, epoch_length)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load metadata
    print(f"Loading metadata from {METADATA_FILE} ...")
    meta_df = pd.read_csv(METADATA_FILE)
    print(f"  {len(meta_df)} subjects in metadata")
    print(f"  Class distribution: {dict(meta_df['label'].value_counts())}\n")

    all_features = []  # list of dicts, one per epoch
    skipped = 0

    print(f"Segmenting into {EPOCH_DURATION_SEC}-sec epochs and extracting features ...\n")

    t_start = time.time()
    for i, row in enumerate(meta_df.itertuples(), 1):
        sid = row.subject_id
        label = row.label
        npy_path = os.path.join(CLEANED_DIR, f"{sid}.npy")

        if not os.path.exists(npy_path):
            print(f"  [{i:3d}/{len(meta_df)}] {sid}: MISSING .npy, skipped")
            skipped += 1
            continue

        # Load cleaned signal
        signal = np.load(npy_path)  # shape: (19, n_samples)

        # Segment into 30-sec epochs
        epochs = segment_subject(signal)
        n_epochs = epochs.shape[0]

        if n_epochs == 0:
            print(f"  [{i:3d}/{len(meta_df)}] {sid}: recording too short for any epoch, skipped")
            skipped += 1
            continue

        # Extract features using mne-features
        # Input must be shape (n_epochs, n_channels, n_samples), in volts
        # Our data is in microvolts -> convert to volts for mne-features
        epochs_volts = epochs * 1e-6

        try:
            feat = extract_features(
                epochs_volts,
                sfreq=SAMPLING_RATE,
                selected_funcs=FEATURE_FUNCS,
                funcs_params=FEATURE_PARAMS,
                n_jobs=1,
                return_as_df=True,
            )
        except Exception as e:
            print(f"  [{i:3d}/{len(meta_df)}] {sid}: feature extraction FAILED -> {e}")
            skipped += 1
            continue

        # feat is a DataFrame: rows = epochs, cols = MultiIndex (feature, channel)
        # Flatten the columns to single-level: "feature_channel"
        feat.columns = [f"{f}_{c}" for f, c in feat.columns]

        # Add metadata columns
        feat.insert(0, "subject_id", sid)
        feat.insert(1, "label", label)
        feat.insert(2, "epoch_idx", np.arange(n_epochs))

        all_features.append(feat)

        if i % 20 == 0 or i == len(meta_df):
            elapsed = time.time() - t_start
            print(f"  [{i:3d}/{len(meta_df)}] {sid}: {n_epochs} epochs, "
                  f"{feat.shape[1] - 3} features. Elapsed: {elapsed:.1f}s")

    if not all_features:
        print("\n[ERROR] No features extracted from any subject. Exiting.")
        return

    # Concatenate all subjects
    print(f"\nConcatenating {len(all_features)} subjects' feature DataFrames...")
    master_df = pd.concat(all_features, ignore_index=True)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "eeg_features.csv")
    master_df.to_csv(out_path, index=False)

    # Summary
    n_features = master_df.shape[1] - 3  # minus subject_id, label, epoch_idx
    n_epochs_total = len(master_df)
    n_subjects = master_df["subject_id"].nunique()

    print(f"\n{'='*70}")
    print(f"DONE. Feature extraction complete.")
    print(f"Saved to: {out_path}")
    print(f"Shape: {master_df.shape}  (epochs x columns)")
    print(f"Total epochs: {n_epochs_total}")
    print(f"Subjects processed: {n_subjects} / {len(meta_df)}")
    print(f"Subjects skipped: {skipped}")
    print(f"Features per epoch: {n_features}")
    print(f"Mean epochs per subject: {n_epochs_total / n_subjects:.1f}")
    print(f"\nClass distribution (epochs):")
    print(master_df["label"].value_counts().to_string())
    print(f"\nClass distribution (subjects):")
    print(master_df.groupby("subject_id")["label"].first().value_counts().to_string())
    print(f"{'='*70}")


if __name__ == "__main__":
    main()