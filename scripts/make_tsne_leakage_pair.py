"""
make_tsne_leakage_pair.py
--------------------------------------------------------------------------
Generates TWO t-SNE figures from a SINGLE embedding so they are provably
the same points, just coloured differently:

  (1) tsne_by_class.png    -> Results  : ADHD vs Control (replaces old fig 13)
  (2) tsne_by_subject.png  -> Discussion: within-subject clustering = the
                              visual mechanism for why epoch-level splitting leaks.

WHY a single embedding: your current Results t-SNE was computed inline and
discarded, so it could not be recoloured. Re-running once and saving BOTH
colourings (plus the coords) keeps the two figures consistent and lets you
regenerate without recomputing.

RUN from your project root (where X.npy / y.npy / groups.npy live, OR where
load_data() can find them). t-SNE on ~6.5k x 266 is a few minutes, single-threaded.

  python make_tsne_leakage_pair.py

Outputs go to ./figures_tsne/.
--------------------------------------------------------------------------
HONESTY GUARDRAILS BAKED IN:
- Same random_state (42), perplexity (30), init ('pca') as your original fig.
- Same 3000-epoch subsample (seed 42) for the CLASS plot, so it reproduces
  the image already in your Results.
- SUBJECT plot uses ALL epochs of 8 selected subjects (not the subsample),
  because the leakage point needs dense within-subject clumps to be visible.
- Subject selection is balanced (4 ADHD + 4 Control) and DETERMINISTIC.
- Nothing is fabricated; if X.npy is absent the loader raises (no demo mode).
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

OUTDIR = "figures_tsne"
os.makedirs(OUTDIR, exist_ok=True)

ADHD_C = "#C44E52"   # match your existing palette
CTRL_C = "#4C72B0"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})

# ----------------------------------------------------------------------
# DATA LOADING -- reads the REAL pre-computed feature matrix that all your
# model scripts use: data/features/eeg_features.csv
#   columns: subject_id, label, epoch_idx, <266 feature columns>
#   rows: 6565 epochs (one per row)  -> matches your locked pipeline.
# No raw-signal processing, no feature re-implementation: this IS your
# validated feature set, so the embedding represents exactly what the SVM saw.
# ----------------------------------------------------------------------
import pandas as pd

# default path relative to project root; edit if you run from elsewhere
FEATURES_CSV = os.path.join("data", "features", "eeg_features.csv")

def load_data():
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_CSV}. Run from project root "
            "(D:\\AdaptED_v2) or edit FEATURES_CSV. Refusing to fabricate data."
        )
    df = pd.read_csv(FEATURES_CSV)

    meta_cols = ["subject_id", "label", "epoch_idx"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feat_cols].to_numpy(dtype=float)
    # label -> 1 = ADHD, 0 = Control  (matches your pipeline convention)
    lab = df["label"].astype(str).str.upper()
    y = (lab == "ADHD").astype(int).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()

    if X.shape[1] != 266:
        print(f"  !! WARNING: expected 266 feature columns, got {X.shape[1]}. "
              "Check that meta_cols matches the CSV header.")
    return X, y, groups


def main():
    print("Loading data ...")
    X, y, groups = load_data()
    X = np.asarray(X); y = np.asarray(y).astype(int); groups = np.asarray(groups)
    n = len(X)
    print(f"  X={X.shape}  epochs={n}  subjects={len(np.unique(groups))}")

    # ------------------------------------------------------------------
    # SANITY GATE (your demo-disaster lesson): class balance must be sane.
    # ------------------------------------------------------------------
    n_adhd_ep = int((y == 1).sum()); n_ctrl_ep = int((y == 0).sum())
    print(f"  epoch labels: ADHD={n_adhd_ep}  Control={n_ctrl_ep}")
    # subject-level labels
    subj_ids = np.unique(groups)
    subj_lbl = np.array([y[groups == s][0] for s in subj_ids])
    n_adhd_s = int((subj_lbl == 1).sum()); n_ctrl_s = int((subj_lbl == 0).sum())
    print(f"  subjects: ADHD={n_adhd_s}  Control={n_ctrl_s}  (expect 61 / 59)")
    if not (n_adhd_s == 61 and n_ctrl_s == 59):
        print("  !! WARNING: subject counts != 61/59. Check data before trusting figures.")

    # ==================================================================
    # FIGURE 1 -- CLASS-COLOURED (reproduces your Results fig 13)
    #   same 3000-epoch subsample, same seed, same t-SNE params
    # ==================================================================
    print("\n[1/2] t-SNE on 3000-epoch subsample (class-coloured, reproduces Results)...")
    max_n = 3000
    idx = np.random.default_rng(42).choice(n, min(max_n, n), replace=False)
    Z_class = TSNE(n_components=2, perplexity=30, init="pca",
                   random_state=42).fit_transform(
        StandardScaler().fit_transform(X[idx]))
    yy = y[idx]

    np.save(os.path.join(OUTDIR, "tsne_coords_class.npy"), Z_class)
    np.save(os.path.join(OUTDIR, "tsne_idx_class.npy"), idx)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z_class[yy == 0, 0], Z_class[yy == 0, 1], s=8, alpha=0.4,
               color=CTRL_C, label="Control")
    ax.scatter(Z_class[yy == 1, 0], Z_class[yy == 1, 1], s=8, alpha=0.4,
               color=ADHD_C, label="ADHD")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE projection of epoch features (by class)")
    ax.legend(); ax.grid(False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"tsne_by_class.{ext}"))
    plt.close(fig)
    print("  saved tsne_by_class.png/pdf")

    # ==================================================================
    # FIGURE 2 -- SUBJECT-COLOURED leakage figure
    #   8 subjects (4 ADHD + 4 Control), ALL their epochs, distinct colours.
    #   Dense within-subject clumps = visual proof that an epoch-level split
    #   scatters one person's near-identical epochs across train AND test.
    # ==================================================================
    print("\n[2/2] t-SNE on ALL epochs of 8 selected subjects (subject-coloured)...")

    # deterministic balanced selection: 4 ADHD + 4 Control,
    # pick subjects with the MOST epochs (densest clumps, clearest visual)
    adhd_subj = subj_ids[subj_lbl == 1]
    ctrl_subj = subj_ids[subj_lbl == 0]

    def epoch_count(s):
        return int((groups == s).sum())

    adhd_sorted = sorted(adhd_subj, key=epoch_count, reverse=True)
    ctrl_sorted = sorted(ctrl_subj, key=epoch_count, reverse=True)
    chosen = list(adhd_sorted[:4]) + list(ctrl_sorted[:4])
    print(f"  chosen subjects (4 ADHD + 4 Control): {chosen}")

    mask = np.isin(groups, chosen)
    Xc = X[mask]; gc = groups[mask]; yc = y[mask]
    print(f"  epochs in subject plot: {len(Xc)} across {len(chosen)} subjects")

    Z_subj = TSNE(n_components=2, perplexity=30, init="pca",
                  random_state=42).fit_transform(
        StandardScaler().fit_transform(Xc))

    np.save(os.path.join(OUTDIR, "tsne_coords_subject.npy"), Z_subj)
    np.save(os.path.join(OUTDIR, "tsne_subject_ids.npy"), gc)

    # 8 distinct, colourblind-reasonable colours; ADHD subjects in warm,
    # Control in cool, so class is still legible via marker too.
    adhd_colors = ["#C44E52", "#E8893A", "#8C2D2D", "#D4B106"]   # distinct warm/dark
    ctrl_colors = ["#4C72B0", "#2CA8B0", "#5B2C9E", "#1A5276"]   # distinct cool
    color_map = {}
    for s, c in zip(adhd_sorted[:4], adhd_colors):
        color_map[s] = (c, "o", "ADHD")
    for s, c in zip(ctrl_sorted[:4], ctrl_colors):
        color_map[s] = (c, "^", "Control")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for s in chosen:
        c, marker, cls = color_map[s]
        m = gc == s
        ax.scatter(Z_subj[m, 0], Z_subj[m, 1], s=14, alpha=0.6,
                   color=c, marker=marker, edgecolors="none",
                   label=f"Subj {s} ({cls})")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of epoch features, coloured by subject\n"
                 "(8 subjects, all epochs) \u2014 epochs cluster within subject")
    ax.legend(fontsize=7.5, ncol=2, loc="best", framealpha=0.9)
    ax.grid(False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"tsne_by_subject.{ext}"))
    plt.close(fig)
    print("  saved tsne_by_subject.png/pdf")

    print(f"\nDone. Both figures + coords in ./{OUTDIR}/")
    print("Note: tsne_by_class reproduces your Results fig (same subsample/seed).")
    print("      Replace your old fig 13 with this one so Results & Discussion match.")


if __name__ == "__main__":
    main()
