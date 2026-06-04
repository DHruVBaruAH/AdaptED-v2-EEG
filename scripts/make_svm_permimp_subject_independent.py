

import os, re, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance


FEATURES_CSV = "data/features/eeg_features.csv" 
OUTDIR       = "."
TOP          = 20
N_SPLITS     = 5
N_REPEATS_CV = 10          
N_PERM       = 10          
NAVY         = "#1F3A5F"


CHANNELS = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
            "F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"]
BANDS = ["delta","theta","alpha","beta","gamma"]
_STEM = {"app_entropy":"approx. entropy","std":"std","ptp_amp":"peak-to-peak",
         "kurtosis":"kurtosis","skewness":"skewness","katz_fd":"Katz FD",
         "hjorth_mobility":"Hjorth mob.","hjorth_complexity":"Hjorth comp.","mean":"mean"}
def _ch(n):  n=int(n);  return CHANNELS[n] if 0<=n<len(CHANNELS) else f"ch{n}"
def _band(b):b=int(b);  return BANDS[b]   if 0<=b<len(BANDS)    else f"band{b}"
def relabel(name):
    m = re.match(r"pow_freq_bands_ch(\d+)_band(\d+)", name)
    if m: return f"{_ch(m.group(1))} {_band(m.group(2))} power"
    m = re.match(r"([a-z_]+?)_ch(\d+)$", name)
    if m: return f"{_ch(m.group(2))} {_STEM.get(m.group(1), m.group(1))}"
    return name

def main():
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Cannot find {FEATURES_CSV} — set the path at the top.")
    df = pd.read_csv(FEATURES_CSV)
    meta = [c for c in ["subject_id","label","epoch_idx"] if c in df.columns]
    feat_names = [c for c in df.columns if c not in meta]
    X = df[feat_names].to_numpy(dtype=float)
    y = (df["label"].astype(str).str.upper() == "ADHD").astype(int).to_numpy()
    groups = df["subject_id"].to_numpy()
    n_feat = X.shape[1]
    print(f"Loaded {X.shape[0]} epochs x {n_feat} features; "
          f"ADHD={y.sum()}, Control={(y==0).sum()}, subjects={len(np.unique(groups))}")

    # StratifiedGroupKFold keeps subjects whole AND balances classes per fold
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        def make_cv(seed): return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    except Exception:
        from sklearn.model_selection import GroupKFold
        print("note: StratifiedGroupKFold unavailable; falling back to GroupKFold (no per-fold class balancing).")
        def make_cv(seed): return GroupKFold(n_splits=N_SPLITS)

    all_imp = []            # collect per-fold mean importance vectors
    fold_aucs = []
    t0 = time.time()
    fit_count = 0
    for rep in range(N_REPEATS_CV):
        cv = make_cv(seed=42 + rep)
        for k, (tr, te) in enumerate(cv.split(X, y, groups)):
            pipe = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=10, gamma=0.001, probability=False, random_state=42),
            )
            pipe.fit(X[tr], y[tr])
            # need both classes in test for AUC (StratifiedGroupKFold ensures this)
            if len(np.unique(y[te])) < 2:
                continue
            r = permutation_importance(pipe, X[te], y[te], scoring="roc_auc",
                                       n_repeats=N_PERM, random_state=42, n_jobs=-1)
            all_imp.append(r.importances_mean)
            # held-out AUC for a sanity read
            from sklearn.metrics import roc_auc_score
            try:
                dec = pipe.decision_function(X[te])
                fold_aucs.append(roc_auc_score(y[te], dec))
            except Exception:
                pass
            fit_count += 1
            el = time.time() - t0
            print(f"  rep {rep+1}/{N_REPEATS_CV} fold {k+1}/{N_SPLITS}  "
                  f"(fit {fit_count})  elapsed {el/60:.1f} min")

    imp_mat = np.vstack(all_imp)              # (folds*repeats, n_feat)
    imp_mean = imp_mat.mean(axis=0)
    imp_std  = imp_mat.std(axis=0)
    if fold_aucs:
        print(f"\nMean held-out AUC across folds: {np.mean(fold_aucs):.4f} "
              f"(SHOULD be near your LOSO 0.78; if ~0.95 STOP - leakage)")

    order = np.argsort(imp_mean)[::-1][:TOP][::-1]
    labels = [relabel(feat_names[i]) for i in order]
    print("\nTop features (subject-independent SVM permutation importance):")
    for i in order[::-1]:
        print(f"  {relabel(feat_names[i]):28s}  {imp_mean[i]:+.5f} ± {imp_std[i]:.5f}")

    fig, ax = plt.subplots(figsize=(7.8, 6.5), dpi=300)
    ax.barh(range(len(order)), imp_mean[order], xerr=imp_std[order],
            color=NAVY, ecolor="#999999", capsize=2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Permutation importance (mean AUC drop, subject-independent)")
    ax.set_title("SVM permutation importance (subject-independent) — descriptive")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUTDIR, "14c_svm_permimp_subject_independent.png")
    pdf = os.path.join(OUTDIR, "14c_svm_permimp_subject_independent.pdf")
    fig.savefig(png, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min over {fit_count} SVM fits")
    print(f"Saved:\n  {os.path.abspath(png)}\n  {os.path.abspath(pdf)}")

if __name__ == "__main__":
    main()