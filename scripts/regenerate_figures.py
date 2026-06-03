"""
Regenerate REAL-data figures, overwriting the DEMO-contaminated ones in place.

Source of truth:
  * Performance figures (07 confusion, 08 ROC, 10 per-subject) come ONLY from the
    saved subject-level out-of-fold predictions in results/loso_predictions.csv.
    No fresh LOSO loop is run here.
  * 01 class dist, 04 band power, 11 PCA scatter, 12 PCA scree, 13 t-SNE,
    14 feature importance come from the real epoch features in
    data/features/eeg_features.csv.

Styling (rcParams, colours, sizes, DPI, filenames) is copied verbatim from
make_figures.py so every output is a drop-in replacement.

NO DEMO FALLBACK: if a real file is missing, this raises and stops.
A hard sanity gate (SVM 89/120, 74.17%, AUC 0.7813, CM [[42,17],[14,47]]) and a
per-model confusion-matrix orientation check run BEFORE any figure is saved.
"""

import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# STYLE  (identical to make_figures.py)
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "figures")
BACKUP = os.path.join(OUTDIR, "_demo_backup")
PRED_CSV = os.path.join(ROOT, "results", "loso_predictions.csv")
FEAT_CSV = os.path.join(ROOT, "data", "features", "eeg_features.csv")

FS = 128
ADHD_C = "#C44E52"
CTRL_C = "#4C72B0"
NAVY = "#1F3A5F"
TEAL = "#0F6E56"
AMBER = "#BA7517"

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

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved {name}")

# Display order / names identical to get_models() insertion order in make_figures.py
MODEL_ORDER = ["SVM", "Logistic Regression", "Random Forest", "XGBoost"]
# Map display name -> CSV 'algorithm' value
CSV_ALGO = {
    "SVM": "SVM",
    "Logistic Regression": "LogisticRegression",
    "Random Forest": "RandomForest",
    "XGBoost": "XGBoost",
}

# ----------------------------------------------------------------------
# DATA LOADING  --  REAL ONLY, NO DEMO FALLBACK
# ----------------------------------------------------------------------
def load_predictions():
    if not os.path.exists(PRED_CSV):
        raise FileNotFoundError(f"REAL predictions not found: {PRED_CSV}. Refusing to use synthetic data.")
    df = pd.read_csv(PRED_CSV)
    need = {"subject_id", "true_label", "algorithm", "mean_prob_adhd", "predicted_label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"loso_predictions.csv missing columns: {missing}")
    return df

def load_features():
    if not os.path.exists(FEAT_CSV):
        raise FileNotFoundError(f"REAL features not found: {FEAT_CSV}. Refusing to use synthetic data.")
    df = pd.read_csv(FEAT_CSV)
    meta = ["subject_id", "label", "epoch_idx"]
    feat_names = [c for c in df.columns if c not in meta]
    X = df[feat_names].values.astype(float)
    y = (df["label"].values == "ADHD").astype(int)   # ADHD=1, Control=0
    groups = df["subject_id"].values
    if X.shape[1] != 266:
        raise ValueError(f"expected 266 features, got {X.shape[1]}")
    if not np.isfinite(X).all():
        raise ValueError("non-finite values in real feature matrix")
    return X, y, groups, feat_names

# ----------------------------------------------------------------------
# Subject-level arrays per model, built from the saved CSV (Control=0, ADHD=1)
# ----------------------------------------------------------------------
def model_arrays(df, display_name):
    algo = CSV_ALGO[display_name]
    s = df[df["algorithm"] == algo]
    if len(s) != 120:
        raise ValueError(f"{display_name}: expected 120 subject rows, got {len(s)}")
    yt = (s["true_label"].values == "ADHD").astype(int)
    yp = (s["predicted_label"].values == "ADHD").astype(int)
    prob = s["mean_prob_adhd"].values.astype(float)
    ids = s["subject_id"].values
    return yt, yp, prob, ids

def cm_oriented(yt, yp):
    """Confusion matrix with rows=true[Control,ADHD], cols=pred[Control,ADHD]."""
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    # Independent manual cross-check of orientation (halt if transposed).
    TN = int(((yt == 0) & (yp == 0)).sum())
    FP = int(((yt == 0) & (yp == 1)).sum())
    FN = int(((yt == 1) & (yp == 0)).sum())
    TP = int(((yt == 1) & (yp == 1)).sum())
    manual = np.array([[TN, FP], [FN, TP]])
    if not np.array_equal(cm, manual):
        raise AssertionError(
            f"Confusion-matrix orientation mismatch: sklearn={cm.tolist()} "
            f"manual[[TN,FP],[FN,TP]]={manual.tolist()} — refusing to save a transposed matrix."
        )
    return cm

# ----------------------------------------------------------------------
# HARD SANITY GATE  (runs before any figure is saved)
# ----------------------------------------------------------------------
def sanity_gate(df):
    yt, yp, prob, _ = model_arrays(df, "SVM")
    correct = int((yt == yp).sum())
    acc = 100.0 * correct / 120
    auc_v = roc_auc_score(yt, prob)
    cm = cm_oriented(yt, yp)
    exp_cm = np.array([[42, 17], [14, 47]])
    ok = (correct == 89 and abs(acc - 74.17) < 0.01 and abs(auc_v - 0.7813) < 1e-4
          and np.array_equal(cm, exp_cm))
    if not ok:
        raise AssertionError(
            "SVM SANITY GATE FAILED — halting, no figures written.\n"
            f"  correct={correct}/120 (want 89), acc={acc:.2f}% (want 74.17), "
            f"AUC={auc_v:.4f} (want 0.7813), CM={cm.tolist()} (want [[42,17],[14,47]])"
        )
    # Orientation check for every model too.
    for name in MODEL_ORDER:
        a, b, _, _ = model_arrays(df, name)
        cm_oriented(a, b)
    print(f"  SANITY GATE PASSED: SVM {correct}/120, {acc:.2f}%, AUC {auc_v:.4f}, CM {cm.tolist()}")

# ----------------------------------------------------------------------
# FIGURES  (code identical to make_figures.py, fed with real data)
# ----------------------------------------------------------------------
def fig_class_distribution(y, groups):
    subj_lbl = np.array([y[groups == s][0] for s in np.unique(groups)])
    n_adhd, n_ctrl = int((subj_lbl == 1).sum()), int((subj_lbl == 0).sum())
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["ADHD", "Control"], [n_adhd, n_ctrl], color=[ADHD_C, CTRL_C], width=0.6)
    for b, v in zip(bars, [n_adhd, n_ctrl]):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Number of subjects"); ax.set_title("Subject class distribution")
    ax.set_ylim(0, max(n_adhd, n_ctrl)*1.15)
    save(fig, "01_class_distribution")

def fig_band_power(X, y, feat_names):
    # Real band columns: pow_freq_bands_ch{ch}_band{0..4} -> delta..gamma
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    Xs = StandardScaler().fit_transform(X)   # standardise band power across epochs
    means_adhd, means_ctrl = [], []
    for i in range(5):
        cols = [j for j, n in enumerate(feat_names) if n.endswith(f"_band{i}") and "pow_freq_bands" in n]
        if not cols:
            raise ValueError(f"no band columns matched band{i} ({bands[i]})")
        means_adhd.append(Xs[y == 1][:, cols].mean())
        means_ctrl.append(Xs[y == 0][:, cols].mean())
    x = np.arange(len(bands)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x-w/2, means_adhd, w, label="ADHD", color=ADHD_C)
    ax.bar(x+w/2, means_ctrl, w, label="Control", color=CTRL_C)
    ax.set_xticks(x); ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_ylabel("Mean relative band power (standardised)")
    ax.set_title("Band-power comparison by group"); ax.legend()
    save(fig, "04_band_power_comparison")

def fig_confusion(y_true, y_pred, model_name):
    cm = cm_oriented(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > cm.max()/2 else NAVY, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Control", "ADHD"]); ax.set_yticklabels(["Control", "ADHD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {model_name} (LOSO)")
    ax.grid(False)
    save(fig, f"07_confusion_{model_name.replace(' ', '_')}")

def fig_roc(loso_results):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, (yt, yp, prob, ids) in loso_results.items():
        fpr, tpr, _ = roc_curve(yt, prob); a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (subject-level, LOSO)"); ax.legend(loc="lower right", fontsize=9)
    save(fig, "08_roc_curves")

def fig_per_subject_accuracy(loso_results):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(loso_results); correct_counts = []
    for name in names:
        yt, yp, prob, ids = loso_results[name]
        correct_counts.append(int((yt == yp).sum()))
    bars = ax.bar(names, correct_counts, color=[NAVY, TEAL, AMBER, "#7A4FB7"][:len(names)], width=0.6)
    for b, v in zip(bars, correct_counts):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v}/120", ha="center", fontweight="bold", fontsize=9)
    ax.set_ylabel("Subjects correctly classified (of 120)")
    ax.set_title("Subject-level correct classifications under LOSO")
    ax.set_ylim(0, 120)
    save(fig, "10_per_subject_correct")

def fig_pca_scatter(X, y):
    Z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=8, alpha=0.4, color=CTRL_C, label="Control")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=8, alpha=0.4, color=ADHD_C, label="ADHD")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA projection of epoch features"); ax.legend()
    save(fig, "11_pca_scatter")

def fig_pca_scree(X):
    p = PCA().fit(StandardScaler().fit_transform(X))
    cum = np.cumsum(p.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(np.arange(1, len(cum)+1), cum, color=NAVY, lw=1.5)
    for thr, c in [(0.90, AMBER), (0.95, ADHD_C)]:
        k = int(np.argmax(cum >= thr)) + 1
        ax.axhline(thr, ls="--", color=c, lw=1); ax.axvline(k, ls=":", color=c, lw=1)
        ax.text(k+1, thr-0.05, f"{int(thr*100)}% at {k} comp", color=c, fontsize=9)
    ax.set_xlabel("Number of components"); ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA explained variance"); ax.set_ylim(0, 1.02)
    save(fig, "12_pca_scree")

def fig_tsne_scatter(X, y, max_n=3000):
    idx = np.random.default_rng(42).choice(len(X), min(max_n, len(X)), replace=False)
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(
        StandardScaler().fit_transform(X[idx]))
    yy = y[idx]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z[yy == 0, 0], Z[yy == 0, 1], s=8, alpha=0.4, color=CTRL_C, label="Control")
    ax.scatter(Z[yy == 1, 0], Z[yy == 1, 1], s=8, alpha=0.4, color=ADHD_C, label="ADHD")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE projection of epoch features"); ax.legend(); ax.grid(False)
    save(fig, "13_tsne_scatter")

def fig_feature_importance(X, y, feat_names, top=20):
    Xs = StandardScaler().fit_transform(X)
    if HAS_XGB:
        mdl = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            eval_metric="logloss", random_state=42, verbosity=0).fit(Xs, y)
        title = "Feature importance (XGBoost gain) - descriptive"
    else:
        mdl = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42).fit(Xs, y)
        title = "Feature importance (Random Forest) - descriptive"
    imp = mdl.feature_importances_
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.barh(range(len(order)), imp[order], color=TEAL)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([feat_names[i] for i in order], fontsize=8)
    ax.set_xlabel("Importance"); ax.set_title(title)
    save(fig, "14_feature_importance")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
REGEN_FILES = [
    "01_class_distribution", "04_band_power_comparison",
    "07_confusion_SVM", "07_confusion_Logistic_Regression",
    "07_confusion_Random_Forest", "07_confusion_XGBoost",
    "08_roc_curves", "10_per_subject_correct",
    "11_pca_scatter", "12_pca_scree", "13_tsne_scatter", "14_feature_importance",
]

def backup_existing():
    os.makedirs(BACKUP, exist_ok=True)
    n = 0
    for name in REGEN_FILES:
        for ext in ("png", "pdf"):
            src = os.path.join(OUTDIR, f"{name}.{ext}")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(BACKUP, f"{name}.{ext}"))
                n += 1
    print(f"  backed up {n} files -> {BACKUP}")

def main():
    print("Loading REAL data...")
    df = load_predictions()
    X, y, groups, feat_names = load_features()
    print(f"  predictions: {len(df)} rows, {df.algorithm.nunique()} models")
    print(f"  features: X={X.shape}, subjects={len(np.unique(groups))}, features={len(feat_names)}")

    print("Running hard sanity gate (before any save)...")
    sanity_gate(df)

    print("Backing up current (DEMO) figures...")
    backup_existing()

    # subject-level arrays from CSV, in canonical model order
    loso_results = {name: model_arrays(df, name) for name in MODEL_ORDER}

    print("Regenerating feature-based figures (eeg_features.csv):")
    fig_class_distribution(y, groups)
    fig_band_power(X, y, feat_names)
    fig_pca_scatter(X, y)
    fig_pca_scree(X)
    fig_tsne_scatter(X, y)
    fig_feature_importance(X, y, feat_names)

    print("Regenerating performance figures (loso_predictions.csv):")
    for name in MODEL_ORDER:
        yt, yp, prob, ids = loso_results[name]
        fig_confusion(yt, yp, name)
    fig_roc(loso_results)
    fig_per_subject_accuracy(loso_results)

    # ---- verify table ----
    print("\n=== VERIFY TABLE (from results/loso_predictions.csv) ===")
    print(f"{'model':22s} {'correct':>9s} {'acc%':>7s} {'AUC':>8s}   CM rows=true[Ctrl,ADHD] cols=pred[Ctrl,ADHD]")
    for name in MODEL_ORDER:
        yt, yp, prob, ids = loso_results[name]
        cm = cm_oriented(yt, yp)
        correct = int((yt == yp).sum())
        acc = 100.0 * correct / 120
        auc_v = roc_auc_score(yt, prob)
        (TN, FP), (FN, TP) = cm
        print(f"{name:22s} {correct:>5d}/120 {acc:>7.2f} {auc_v:>8.4f}   "
              f"TN={TN} FP={FP} FN={FN} TP={TP}")
    print(f"\nDone. Regenerated {len(REGEN_FILES)} figures (PNG+PDF) in {OUTDIR}/")

if __name__ == "__main__":
    main()
