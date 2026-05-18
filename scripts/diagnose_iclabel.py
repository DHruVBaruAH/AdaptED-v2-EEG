"""
Diagnostic: inspect what ICLabel is actually labeling components as.
Runs on ONE subject only. Prints full label + confidence for each component.

Run from D:/AdaptED_v2/:
    python scripts/diagnose_iclabel.py
"""
import warnings
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

RAW_CSV_PATH = "D:/ADHD_personalized/eeg_data/adhdata.csv"
CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

# Pick one subject to inspect
TARGET_SUBJECT = "v10p"  # change if needed

print(f"Loading data, picking subject {TARGET_SUBJECT}...")
df = pd.read_csv(RAW_CSV_PATH)
sub_df = df[df["ID"] == TARGET_SUBJECT].reset_index(drop=True)
print(f"  Got {len(sub_df)} samples, label={sub_df['Class'].iloc[0]}")

# Build MNE Raw
data = sub_df[CHANNELS].to_numpy(dtype=np.float64).T * 1e-6
info = mne.create_info(ch_names=CHANNELS, sfreq=128, ch_types=["eeg"]*19)
raw = mne.io.RawArray(data, info, verbose=False)
raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                match_case=False, on_missing="ignore")
raw.set_eeg_reference("average", verbose=False)

# Bandpass + notch
raw_filt = raw.copy().filter(l_freq=0.5, h_freq=40, fir_design="firwin", verbose=False)
try:
    raw_filt = raw_filt.notch_filter(freqs=[50], fir_design="firwin", verbose=False)
except Exception:
    pass

# Fit ICA
print("\nFitting ICA...")
ica = ICA(n_components=19, method="infomax",
          fit_params=dict(extended=True), random_state=42,
          max_iter="auto", verbose=False)
ica.fit(raw_filt, verbose=False)

# ICLabel
print("Running ICLabel...\n")
labels_dict = label_components(raw_filt, ica, method="iclabel")
ic_labels = labels_dict["labels"]
ic_probs = labels_dict["y_pred_proba"]

# Print everything
print("="*70)
print(f"ICLabel output for subject {TARGET_SUBJECT}")
print("="*70)
print(f"{'IC#':>4} | {'Label':<20} | {'Confidence':>10}")
print("-"*70)
for i, (lbl, prob) in enumerate(zip(ic_labels, ic_probs)):
    print(f"{i:>4} | {lbl:<20} | {prob:>10.3f}")
print("="*70)

# Summary by label
print("\nLabel distribution:")
from collections import Counter
label_counts = Counter(ic_labels)
for lbl, count in label_counts.most_common():
    print(f"  {lbl}: {count}")

print(f"\nConfidence stats:")
print(f"  Max confidence overall: {max(ic_probs):.3f}")
print(f"  Min confidence overall: {min(ic_probs):.3f}")
print(f"  Mean confidence: {np.mean(ic_probs):.3f}")

# How many would be removed at various thresholds
print(f"\nIf we removed all NON-brain components above threshold T:")
for T in [0.3, 0.4, 0.5, 0.6, 0.7]:
    n_rem = sum(1 for lbl, prob in zip(ic_labels, ic_probs)
                if lbl != "brain" and prob >= T)
    print(f"  T={T}: would remove {n_rem} of 19 components")

print(f"\nIf we removed only KNOWN ARTIFACT labels above threshold T:")
artifacts = ("muscle artifact", "eye blink", "heart beat", "line noise", "channel noise")
for T in [0.3, 0.4, 0.5, 0.6, 0.7]:
    n_rem = sum(1 for lbl, prob in zip(ic_labels, ic_probs)
                if lbl in artifacts and prob >= T)
    print(f"  T={T}: would remove {n_rem} of 19 components")