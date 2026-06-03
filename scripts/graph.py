"""
fig_tfr_raw_vs_filtered.py  (v2 - shared color scale)
--------------------------------------------------------------------
Two-panel time-frequency figure (raw vs filtered), Garcia-Ponsoda style,
for the AdaptED v2 EEG-ADHD paper.

What it shows:
  (a) RAW signal  -> 50 Hz power-line stripe + energy up to ~64 Hz (Nyquist)
  (b) FILTERED    -> 50 Hz stripe gone, hard cutoff at 40 Hz

v2 change: BOTH panels now share ONE color scale (vmin/vmax derived from the
raw panel), so power is directly comparable between them. A single shared
colorbar is drawn on the right.

Method: Morlet wavelet TFR on one channel from one subject. Mirrors
eeg_preprocessing.py exactly (CHANNELS order, average reference, uV->V).

HONEST SCOPE: demonstrates BAND-PASS FILTERING only, NOT artifact-component
removal (no ICA). Caption it as such, and name the subject as the first in
the dataset (no cherry-picking).

Usage:
  1. set RAW_CSV_PATH below (your adhdata.csv) - use / or \\ , never single \
  2. optionally set SUBJECT_ID (None = first subject in the file)
  3. python fig_tfr_raw_vs_filtered.py
Outputs PNG (300 dpi) + PDF into ./output/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("ERROR")

# ----------------------------------------------------------------------
# CONFIG  -- matches eeg_preprocessing.py
# ----------------------------------------------------------------------
RAW_CSV_PATH = "D:/ADHD_personalized/eeg_data/adhdata.csv"   # <-- set this
SUBJECT_ID   = None          # None = use the first subject in the CSV
CHANNEL      = "Fp1"         # single channel to plot
OUTPUT_DIR   = "output"

SAMPLING_RATE = 128
BANDPASS_LOW  = 0.5
BANDPASS_HIGH = 40.0
NOTCH_FREQ    = 50.0

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

# TFR settings
SECONDS    = 80                      # length of segment to display
FMIN, FMAX = 1.0, 64.0               # up to Nyquist so the >40 Hz band is visible
N_FREQS    = 80
DECIM      = 8                       # downsample TFR in time for a lighter plot


def build_raw(sub_df):
    data = sub_df[CHANNELS].to_numpy(dtype=np.float64).T * 1e-6  # uV -> V
    info = mne.create_info(CHANNELS, SAMPLING_RATE, ch_types=["eeg"] * len(CHANNELS))
    raw = mne.io.RawArray(data, info)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    match_case=False, on_missing="ignore")
    raw.set_eeg_reference("average")
    return raw


def filter_raw(raw):
    out = raw.copy().filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
                            fir_design="firwin")
    try:
        out = out.notch_filter(freqs=[NOTCH_FREQ], fir_design="firwin")
    except Exception:
        pass
    return out


def morlet_tfr(raw, channel, n_seconds):
    """Return (freqs, times, power_dB[freq,time]) for one channel via Morlet TFR."""
    n = min(int(n_seconds * SAMPLING_RATE), raw.n_times)
    sig = raw.get_data(picks=[channel])[0, :n]
    x = sig[np.newaxis, np.newaxis, :]                     # (epochs, chans, times)
    freqs = np.linspace(FMIN, FMAX, N_FREQS)
    n_cycles = np.maximum(freqs / 2.0, 2.0)
    power = mne.time_frequency.tfr_array_morlet(
        x, sfreq=SAMPLING_RATE, freqs=freqs, n_cycles=n_cycles,
        output="power", decim=DECIM,
    )[0, 0]                                                # (freq, time)
    times = np.arange(power.shape[1]) * DECIM / SAMPLING_RATE
    power_db = 10 * np.log10(power + 1e-20)
    return freqs, times, power_db


def plot_panel(ax, freqs, times, power_db, title, vmin, vmax):
    im = ax.pcolormesh(times, freqs, power_db, shading="auto",
                       cmap="jet", vmin=vmin, vmax=vmax)   # SHARED scale
    ax.axhline(BANDPASS_HIGH, color="white", ls="--", lw=1.0, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return im


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_CSV_PATH)
    sid = SUBJECT_ID if SUBJECT_ID is not None else df["ID"].unique()[0]
    sub_df = df[df["ID"] == sid].reset_index(drop=True)
    label = sub_df["Class"].iloc[0]
    print(f"Subject {sid} ({label}), {len(sub_df)} samples")

    raw = build_raw(sub_df)
    filt = filter_raw(raw)

    f1, t1, p1 = morlet_tfr(raw,  CHANNEL, SECONDS)
    f2, t2, p2 = morlet_tfr(filt, CHANNEL, SECONDS)

    # ---- shared color scale, derived from the RAW panel ----
    # robust limits via percentiles so a few extreme cells don't wash it out
    vmax = np.percentile(p1, 99.5)
    vmin = np.percentile(p1, 5.0)
    print(f"Shared color scale: vmin={vmin:.1f} dB, vmax={vmax:.1f} dB")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300, sharey=True)
    plot_panel(axa, f1, t1, p1, f"(a) Raw signal — {CHANNEL}", vmin, vmax)
    im2 = plot_panel(axb, f2, t2, p2, f"(b) Filtered (0.5-40 Hz) — {CHANNEL}", vmin, vmax)

    # one shared colorbar for both panels
    cb = fig.colorbar(im2, ax=(axa, axb), pad=0.02, fraction=0.046)
    cb.set_label("Power (dB)")

    fig.suptitle(f"Time-frequency representation (first subject: {sid}, {label})", y=1.03)

    png = os.path.join(OUTPUT_DIR, "fig_tfr_raw_vs_filtered.png")
    pdf = os.path.join(OUTPUT_DIR, "fig_tfr_raw_vs_filtered.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {png}\n  {pdf}")
    print("Expected: (a) shows a 50 Hz stripe + energy above 40 Hz; "
          "(b) is dark above the dashed 40 Hz line, 50 Hz stripe gone.")


if __name__ == "__main__":
    main()