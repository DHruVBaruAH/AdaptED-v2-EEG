

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")


RAW_CSV_PATH = "D:/ADHD_personalized/eeg_data/adhdata.csv"
OUTPUT_DIR = "data/cleaned"
RESULTS_DIR = "results"

SAMPLING_RATE = 128
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
NOTCH_FREQ = 50.0

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

MIN_SAMPLES = 9250  # matches Esas 2023 (~72 sec at 128 Hz)


def load_raw_csv():
    print(f"Loading {RAW_CSV_PATH} ...")
    df = pd.read_csv(RAW_CSV_PATH)
    print(f"  Total rows: {df.shape[0]:,}, columns: {df.shape[1]}")
    print(f"  Unique subjects: {df['ID'].nunique()}")
    print(f"  Class distribution: {dict(df['Class'].value_counts())}")
    return df


def subject_to_raw(subject_df, subject_id):
    data = subject_df[CHANNELS].to_numpy(dtype=np.float64).T
    data = data * 1e-6  # microvolts -> volts (MNE convention)

    info = mne.create_info(
        ch_names=CHANNELS,
        sfreq=SAMPLING_RATE,
        ch_types=["eeg"] * len(CHANNELS),
    )

    raw = mne.io.RawArray(data, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    raw.set_eeg_reference("average", verbose=False)
    return raw


def preprocess_subject(raw, subject_id):
    """Apply bandpass + notch filtering. ICA disabled - see header docstring."""
    # Bandpass filter (0.5-40 Hz, FIR)
    raw_filt = raw.copy().filter(
        l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
        fir_design="firwin", verbose=False,
    )

    # Notch filter at 50 Hz (power line interference)
    try:
        raw_filt = raw_filt.notch_filter(
            freqs=[NOTCH_FREQ], fir_design="firwin", verbose=False,
        )
    except Exception:
        pass  # notch failure is non-fatal, continue with bandpass only

    return raw_filt


def plot_comparison(raw_original, raw_cleaned, subject_id, output_path):
    fs = int(SAMPLING_RATE)
    n_seconds = 5
    n_samples = fs * n_seconds

    raw_data = raw_original.get_data(picks=["Fp1"])[0, :n_samples] * 1e6
    clean_data = raw_cleaned.get_data(picks=["Fp1"])[0, :n_samples] * 1e6
    t = np.arange(n_samples) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(t, raw_data, color="#c0392b", linewidth=0.8)
    ax1.set_title(f"RAW signal (Fp1, first 5 seconds) - subject {subject_id}")
    ax1.set_ylabel("Amplitude (uV)")
    ax1.grid(alpha=0.3)

    ax2.plot(t, clean_data, color="#27ae60", linewidth=0.8)
    ax2.set_title("FILTERED signal (bandpass 0.5-40 Hz + notch 50 Hz)")
    ax2.set_ylabel("Amplitude (uV)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"    Saved comparison plot -> {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_raw_csv()
    subjects = df["ID"].unique()
    print(f"\nProcessing {len(subjects)} subjects ...")
    print(f"Bandpass: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz, Notch: {NOTCH_FREQ} Hz")
    print(f"ICA: disabled (see script header for justification)\n")

    metadata = []
    plot_done = False

    for i, sid in enumerate(subjects, 1):
        sub_df = df[df["ID"] == sid].reset_index(drop=True)
        label = sub_df["Class"].iloc[0]
        n_samples = len(sub_df)

        if n_samples < MIN_SAMPLES:
            print(f"  [{i:3d}/{len(subjects)}] {sid:8s} ({label:7s}): SKIPPED ({n_samples} samples < {MIN_SAMPLES})")
            continue

        t0 = time.time()
        try:
            raw = subject_to_raw(sub_df, sid)
            raw_clean = preprocess_subject(raw, sid)

            cleaned_data = raw_clean.get_data() * 1e6  # back to microvolts
            out_path = os.path.join(OUTPUT_DIR, f"{sid}.npy")
            np.save(out_path, cleaned_data.astype(np.float32))

            metadata.append({
                "subject_id": sid,
                "label": label,
                "n_samples": cleaned_data.shape[1],
                "duration_sec": cleaned_data.shape[1] / SAMPLING_RATE,
            })

            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(subjects)}] {sid:8s} ({label:7s}): "
                  f"{cleaned_data.shape[1]:6d} samples, {elapsed:.2f}s")

            if not plot_done:
                plot_path = os.path.join(RESULTS_DIR, f"preprocessing_demo_{sid}.png")
                plot_comparison(raw, raw_clean, sid, plot_path)
                plot_done = True

        except Exception as e:
            print(f"  [{i:3d}/{len(subjects)}] {sid:8s}: FAILED -> {e}")

    meta_df = pd.DataFrame(metadata)
    meta_path = os.path.join(OUTPUT_DIR, "_metadata.csv")
    meta_df.to_csv(meta_path, index=False)

    print(f"\n{'='*70}")
    print(f"DONE. Processed {len(metadata)} / {len(subjects)} subjects.")
    print(f"Cleaned signals saved to: {OUTPUT_DIR}/")
    print(f"Metadata saved to: {meta_path}")
    print(f"Total duration: {meta_df['duration_sec'].sum():.1f} sec across all subjects")
    print(f"Mean duration per subject: {meta_df['duration_sec'].mean():.1f} sec")
    print(f"Class distribution after cleaning:")
    print(meta_df["label"].value_counts().to_string())
    print(f"{'='*70}")


if __name__ == "__main__":
    main()