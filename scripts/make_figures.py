"""
AdaptED v2 - Figure generation for the paper
=============================================
Generates every data-driven figure from the EEG ADHD pipeline:
  1.  Class distribution
  2.  Raw multichannel EEG time series
  3.  Preprocessing before/after (PSD)
  4.  Band-power comparison (ADHD vs Control)
  5.  Epoching illustration
  6.  Protocol-gap bar chart  (the centrepiece: 93% leaky vs 74% LOSO)
  7.  LOSO confusion matrix (per model)
  8.  LOSO ROC curves (all models)
  9.  Model comparison bar (Acc / AUC / F1 / Precision / Recall)
  10. Per-subject LOSO accuracy distribution
  11. PCA scatter (2D)
  12. PCA scree / cumulative variance
  13. t-SNE scatter
  14. Feature importance (tree model, descriptive)
  15. SHAP summary (optional, descriptive)
  16. Statistical-significance summary (binomial vs chance)

HONESTY RULES BUILT IN (do not remove):
  * ROC + confusion come ONLY from leave-one-subject-out out-of-fold predictions.
    Never score a model on its own training data - that is the leaky 93% case.
  * LOSO predictions are aggregated to SUBJECT level (mean epoch probability),
    which reproduces the 89/120 = 74.17% subject-level accuracy.
  * Feature importance / SHAP are DESCRIPTIVE (association, not causation).
  * PCA / t-SNE are plotted as-is. They show little separation - that is honest
    and consistent with the modest 74%.

USAGE:
  1. Edit load_data() to point at your real feature matrix.
  2. Run:  python make_figures.py
  3. Figures are written to ./figures/ as 300-dpi PNG + PDF.
  If no data is found it runs in DEMO mode with synthetic data so you can see
  the code works - DEMO figures are NOT your results, they are placeholders.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)
FS = 128                      # sampling rate (Hz)
ADHD_C = "#C44E52"            # ADHD colour
CTRL_C = "#4C72B0"            # control colour
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

# ----------------------------------------------------------------------
# REPORTED NUMBERS (from your paper - used by figures that summarise results)
# ----------------------------------------------------------------------
PROTOCOL_RESULTS = {           # SVM across the three validation schemes
    "Epoch-shuffled\n5-fold (leaky)": {"acc": 93.28, "auc": 0.9805, "leaky": True},
    "Subject-level\n5-fold":          {"acc": 67.50, "auc": 0.7121, "leaky": False},
    "LOSO\n(primary)":                {"acc": 74.17, "auc": 0.7813, "leaky": False},
}
LOSO_TABLE = {                 # Table 2: all four models under LOSO
    "SVM":                 {"acc": 74.17, "auc": 0.7813, "f1": 0.7520, "prec": 0.7344, "rec": 0.7705},
    "Logistic Regression": {"acc": 71.67, "auc": 0.7683, "f1": 0.7424, "prec": 0.6901, "rec": 0.8033},
    "XGBoost":             {"acc": 72.50, "auc": 0.7402, "f1": 0.7519, "prec": 0.6944, "rec": 0.8197},
    "Random Forest":       {"acc": 69.17, "auc": 0.7146, "f1": 0.7413, "prec": 0.6463, "rec": 0.8689},
}
BINOMIAL = {"SVM": 89, "XGBoost": 87, "Logistic Regression": 86, "Random Forest": 83}  # correct / 120

# ----------------------------------------------------------------------
# MODELS (your fixed hyperparameters - set a priori, not tuned on folds)
# ----------------------------------------------------------------------
def get_models():
    m = {
        "SVM": SVC(kernel="rbf", C=10, gamma=0.001, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    }
    if HAS_XGB:
        m["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     eval_metric="logloss", random_state=42, verbosity=0)
    return m

# ----------------------------------------------------------------------
# DATA LOADING  --  EDIT THIS to point at your real data
# ----------------------------------------------------------------------
def load_data():
    """
    Return:
      X            : (n_epochs, n_features)  feature matrix
      y            : (n_epochs,)             labels  (1 = ADHD, 0 = Control), per epoch
      groups       : (n_epochs,)             subject id per epoch  (for LOSO)
      feat_names   : list[str]               length n_features
      raw_example  : (n_channels, n_samples) one raw recording for the time-series fig (or None)
      ch_names     : list[str]               channel names (or None)
    -------------------------------------------------------------------
    REAL DATA: replace the demo block below, e.g.
        X = np.load("D:/AdaptED_v2/features/X.npy")
        y = np.load("D:/AdaptED_v2/features/y.npy")
        groups = np.load("D:/AdaptED_v2/features/groups.npy")
        feat_names = list(np.load("D:/AdaptED_v2/features/feat_names.npy", allow_pickle=True))
    """
    real_X = "X.npy"
    if os.path.exists(real_X):
        X = np.load("X.npy")
        y = np.load("y.npy")
        groups = np.load("groups.npy")
        feat_names = list(np.load("feat_names.npy", allow_pickle=True))
        raw_example = np.load("raw_example.npy") if os.path.exists("raw_example.npy") else None
        ch_names = list(np.load("ch_names.npy", allow_pickle=True)) if os.path.exists("ch_names.npy") else None
        return X, y, groups, feat_names, raw_example, ch_names

    # ---------------- NO DEMO FALLBACK ----------------
    # Real data is missing. Refuse to silently fabricate results: raise instead of
    # generating synthetic DEMO figures that would be mistaken for real findings.
    raise FileNotFoundError(
        f"Real feature data not found (expected '{real_X}' in the working directory). "
        "Refusing to fall back to synthetic DEMO data. Provide the real X.npy/y.npy/"
        "groups.npy/feat_names.npy (or run scripts/regenerate_figures.py, which reads "
        "results/loso_predictions.csv and data/features/eeg_features.csv directly)."
    )

    # ---------------- DEMO MODE (synthetic placeholder) -- DISABLED, kept for reference ----------------
    print("!! DEMO MODE: synthetic data. These figures are NOT your results. !!")
    rng = np.random.default_rng(42)
    n_subj, n_feat = 120, 266
    chans = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
             "F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"]
    bands = ["delta","theta","alpha","beta","gamma"]
    feat_names = []
    for ch in chans:
        feat_names += [f"{ch}_mean", f"{ch}_std", f"{ch}_kurtosis", f"{ch}_skewness", f"{ch}_ptp"]
        feat_names += [f"{ch}_pow_{b}" for b in bands]
        feat_names += [f"{ch}_app_entropy", f"{ch}_katz_fd", f"{ch}_hjorth_mob", f"{ch}_hjorth_comp"]
    feat_names = feat_names[:n_feat]
    while len(feat_names) < n_feat:
        feat_names.append(f"feat_{len(feat_names)}")

    subj_label = np.array([1]*61 + [0]*59)
    rng.shuffle(subj_label)
    X_list, y_list, g_list = [], [], []
    for s in range(n_subj):
        n_ep = rng.integers(40, 70)               # epochs per subject
        base = rng.normal(0, 1, n_feat)
        signal = 0.45 if subj_label[s] == 1 else 0.0   # weak, realistic separation
        for _ in range(n_ep):
            x = base + rng.normal(0, 1, n_feat)
            x[5:60] += signal * rng.normal(1, 0.5, 55)
            X_list.append(x); y_list.append(subj_label[s]); g_list.append(s)
    X = np.array(X_list); y = np.array(y_list); groups = np.array(g_list)
    raw_example = rng.normal(0, 20, (19, FS*10)).cumsum(axis=1)
    raw_example -= raw_example.mean(axis=1, keepdims=True)
    return X, y, groups, feat_names, raw_example, chans

# ----------------------------------------------------------------------
# CORE: leave-one-subject-out, aggregated to subject level
# ----------------------------------------------------------------------
def run_loso(X, y, groups, model):
    """Out-of-fold LOSO. Returns SUBJECT-level true, pred, prob (reproduces 74.17%)."""
    logo = LeaveOneGroupOut()
    subj_true, subj_pred, subj_prob, subj_ids = [], [], [], []
    for tr, te in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])          # fit on TRAIN only (no leakage)
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        mdl = model.__class__(**model.get_params())
        mdl.fit(Xtr, y[tr])
        prob = mdl.predict_proba(Xte)[:, 1]
        subj_prob.append(prob.mean())                 # aggregate epochs -> subject
        subj_pred.append(int(prob.mean() >= 0.5))
        subj_true.append(int(y[te][0]))
        subj_ids.append(groups[te][0])
    return (np.array(subj_true), np.array(subj_pred),
            np.array(subj_prob), np.array(subj_ids))

# ======================================================================
# FIGURE FUNCTIONS
# ======================================================================
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

def fig_raw_timeseries(raw, ch_names, n_show=8, seconds=5):
    if raw is None:
        print("  (skip raw time series - no raw_example provided)"); return
    n_show = min(n_show, raw.shape[0]); n = int(seconds*FS)
    t = np.arange(n)/FS
    fig, ax = plt.subplots(figsize=(9, 6))
    offset = np.nanmax(np.abs(raw[:n_show, :n])) * 1.6
    for i in range(n_show):
        ax.plot(t, raw[i, :n] + i*offset, color=NAVY, lw=0.7)
    ax.set_yticks([i*offset for i in range(n_show)])
    ax.set_yticklabels(ch_names[:n_show] if ch_names else [f"ch{i}" for i in range(n_show)])
    ax.set_xlabel("Time (s)"); ax.set_title("Raw EEG (example subject)")
    ax.grid(False)
    save(fig, "02_raw_timeseries")

def fig_preprocessing_psd(raw, ch_names):
    if raw is None:
        print("  (skip preprocessing PSD - no raw_example provided)"); return
    sig = raw[0]
    # crude band-pass 0.5-40 via FFT mask, to illustrate before/after
    freqs = np.fft.rfftfreq(len(sig), 1/FS)
    F = np.fft.rfft(sig)
    mask = (freqs >= 0.5) & (freqs <= 40)
    filt = np.fft.irfft(F*mask, n=len(sig))
    def psd(x):
        f = np.fft.rfftfreq(len(x), 1/FS); p = np.abs(np.fft.rfft(x))**2
        return f, 10*np.log10(p+1e-12)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    f0, p0 = psd(sig); f1, p1 = psd(filt)
    ax.plot(f0, p0, color="#999999", lw=1, label="Raw")
    ax.plot(f1, p1, color=TEAL, lw=1.3, label="Filtered 0.5-40 Hz")
    ax.axvspan(0.5, 40, color=TEAL, alpha=0.07)
    ax.set_xlim(0, 64); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (dB)")
    ax.set_title("Power spectral density: before vs after filtering"); ax.legend()
    save(fig, "03_preprocessing_psd")

def fig_band_power(X, y, feat_names):
    bands = ["delta","theta","alpha","beta","gamma"]
    means_adhd, means_ctrl = [], []
    for b in bands:
        cols = [i for i,n in enumerate(feat_names) if f"pow_{b}" in n.lower() or n.lower().endswith(b)]
        if not cols:
            print(f"  (band-power fig: no columns matched '{b}' - check feat_names)"); return
        means_adhd.append(X[y==1][:,cols].mean()); means_ctrl.append(X[y==0][:,cols].mean())
    x = np.arange(len(bands)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x-w/2, means_adhd, w, label="ADHD", color=ADHD_C)
    ax.bar(x+w/2, means_ctrl, w, label="Control", color=CTRL_C)
    ax.set_xticks(x); ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_ylabel("Mean relative band power (standardised)")
    ax.set_title("Band-power comparison by group"); ax.legend()
    save(fig, "04_band_power_comparison")

def fig_epoching_illustration():
    fig, ax = plt.subplots(figsize=(9, 2.6))
    t = np.linspace(0, 20, 2000); sig = np.sin(2*np.pi*0.7*t)+0.4*np.random.default_rng(1).normal(size=t.size)
    ax.plot(t, sig, color=NAVY, lw=0.6)
    for k, start in enumerate(np.arange(0, 16, 2.5)):   # 5 s windows, 2.5 s step (50% overlap)
        ax.axvspan(start, start+5, color=AMBER if k%2 else TEAL, alpha=0.12)
    ax.set_xlabel("Time (s)"); ax.set_yticks([])
    ax.set_title("Epoching: 5 s windows, 50% overlap")
    save(fig, "05_epoching_illustration")

def fig_protocol_gap():
    names = list(PROTOCOL_RESULTS); accs = [PROTOCOL_RESULTS[n]["acc"] for n in names]
    cols = [AMBER if PROTOCOL_RESULTS[n]["leaky"] else TEAL for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(names, accs, color=cols, width=0.6)
    for b, v in zip(bars, accs):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.2f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Effect of validation protocol on reported accuracy (SVM)")
    ax.axhline(50, ls="--", color="#999999", lw=1); ax.text(2.4, 52, "chance", color="#777777", fontsize=9)
    save(fig, "06_protocol_gap")

def fig_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=15,
                    color="white" if cm[i,j] > cm.max()/2 else NAVY, fontweight="bold")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Control","ADHD"]); ax.set_yticklabels(["Control","ADHD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {model_name} (LOSO)")
    ax.grid(False)
    save(fig, f"07_confusion_{model_name.replace(' ','_')}")

def fig_roc(loso_results):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, (yt, yp, prob, ids) in loso_results.items():
        fpr, tpr, _ = roc_curve(yt, prob); a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {a:.3f})")
    ax.plot([0,1],[0,1], ls="--", color="#999999", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (subject-level, LOSO)"); ax.legend(loc="lower right", fontsize=9)
    save(fig, "08_roc_curves")

def fig_model_comparison():
    models = list(LOSO_TABLE); metrics = ["acc","auc","f1","prec","rec"]
    labels = ["Accuracy","AUC","F1","Precision","Recall"]
    # normalise accuracy to 0-1 for plotting alongside the others
    data = {m: [(LOSO_TABLE[m]["acc"]/100 if k=="acc" else LOSO_TABLE[m][k]) for k in metrics] for m in models}
    x = np.arange(len(metrics)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.8))
    palette = [NAVY, TEAL, AMBER, "#7A4FB7"]
    for i, m in enumerate(models):
        ax.bar(x + (i-1.5)*w, data[m], w, label=m, color=palette[i%4])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score (accuracy shown as proportion)")
    ax.set_title("LOSO performance across classifiers"); ax.legend(fontsize=9, ncol=2)
    save(fig, "09_model_comparison")

def fig_per_subject_accuracy(loso_results):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(loso_results); correct_counts = []
    for name in names:
        yt, yp, prob, ids = loso_results[name]
        correct_counts.append((yt == yp).sum())
    bars = ax.bar(names, correct_counts, color=[NAVY,TEAL,AMBER,"#7A4FB7"][:len(names)], width=0.6)
    for b, v in zip(bars, correct_counts):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v}/120", ha="center", fontweight="bold", fontsize=9)
    ax.set_ylabel("Subjects correctly classified (of 120)")
    ax.set_title("Subject-level correct classifications under LOSO")
    ax.set_ylim(0, 120)
    save(fig, "10_per_subject_correct")

def fig_pca_scatter(X, y):
    Z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z[y==0,0], Z[y==0,1], s=8, alpha=0.4, color=CTRL_C, label="Control")
    ax.scatter(Z[y==1,0], Z[y==1,1], s=8, alpha=0.4, color=ADHD_C, label="ADHD")
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
    ax.scatter(Z[yy==0,0], Z[yy==0,1], s=8, alpha=0.4, color=CTRL_C, label="Control")
    ax.scatter(Z[yy==1,0], Z[yy==1,1], s=8, alpha=0.4, color=ADHD_C, label="ADHD")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE projection of epoch features"); ax.legend(); ax.grid(False)
    save(fig, "13_tsne_scatter")

def fig_feature_importance(X, y, feat_names, top=20):
    """DESCRIPTIVE: importance from a tree model trained on all data. Association, not causation."""
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

def fig_shap(X, y, feat_names, max_n=1500, top=20):
    """OPTIONAL explainability. Needs:  pip install shap. DESCRIPTIVE only."""
    try:
        import shap
    except Exception:
        print("  (skip SHAP - 'pip install shap' to enable)"); return
    if not HAS_XGB:
        print("  (skip SHAP - XGBoost not installed)"); return
    Xs = StandardScaler().fit_transform(X)
    mdl = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        eval_metric="logloss", random_state=42, verbosity=0).fit(Xs, y)
    idx = np.random.default_rng(42).choice(len(Xs), min(max_n, len(Xs)), replace=False)
    expl = shap.TreeExplainer(mdl)
    sv = expl.shap_values(Xs[idx])
    fig = plt.figure(figsize=(8, 6.5))
    shap.summary_plot(sv, Xs[idx], feature_names=feat_names, max_display=top, show=False)
    plt.title("SHAP summary (XGBoost) - descriptive", fontsize=12)
    save(fig, "15_shap_summary")

def fig_significance():
    models = list(BINOMIAL); correct = [BINOMIAL[m] for m in models]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(models, correct, color=NAVY, width=0.6)
    ax.axhline(60, ls="--", color=ADHD_C, lw=1.2); ax.text(3.1, 61, "chance (60/120)", color=ADHD_C, fontsize=9)
    for b, v in zip(bars, correct):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v}/120\np<0.0001", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Correct (of 120 subjects)"); ax.set_ylim(0, 120)
    ax.set_title("Binomial test vs chance (all models above chance)")
    ax.tick_params(axis="x", labelrotation=15)
    save(fig, "16_significance_binomial")

# ======================================================================
# MAIN
# ======================================================================
def main():
    print("Loading data...")
    X, y, groups, feat_names, raw_example, ch_names = load_data()
    print(f"  X={X.shape}  subjects={len(np.unique(groups))}  features={len(feat_names)}")

    print("Static / descriptive figures:")
    fig_class_distribution(y, groups)
    fig_raw_timeseries(raw_example, ch_names)
    fig_preprocessing_psd(raw_example, ch_names)
    fig_band_power(X, y, feat_names)
    fig_epoching_illustration()
    fig_protocol_gap()
    fig_pca_scatter(X, y)
    fig_pca_scree(X)
    fig_tsne_scatter(X, y)
    fig_feature_importance(X, y, feat_names)
    fig_shap(X, y, feat_names)
    fig_significance()
    fig_model_comparison()

    print("Running LOSO for confusion / ROC / per-subject (this is the slow part)...")
    loso_results = {}
    for name, mdl in get_models().items():
        print(f"  LOSO: {name}")
        yt, yp, prob, ids = run_loso(X, y, groups, mdl)
        loso_results[name] = (yt, yp, prob, ids)
        acc = accuracy_score(yt, yp); a = roc_auc_score(yt, prob)
        print(f"    subject-level acc={acc*100:.2f}%  AUC={a:.4f}  correct={int((yt==yp).sum())}/{len(yt)}")
        fig_confusion(yt, yp, name)
    fig_roc(loso_results)
    fig_per_subject_accuracy(loso_results)
    print(f"\nDone. Figures in ./{OUTDIR}/")

if __name__ == "__main__":
    main()