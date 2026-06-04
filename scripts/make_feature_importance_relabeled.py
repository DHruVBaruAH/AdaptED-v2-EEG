

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


FEATURES_CSV = "data/features/eeg_features.csv"  
OUTDIR       = "."                                
TOP          = 20
TEAL         = "#0F6E56"


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
    if m:
        stem,n = m.group(1), m.group(2)
        return f"{_ch(n)} {_STEM.get(stem, stem)}"
    return name

def main():
    try:
        from xgboost import XGBClassifier
        mdl = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            eval_metric="logloss", random_state=42, verbosity=0)
        title = "Feature importance (XGBoost gain) — descriptive"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        mdl = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        title = "Feature importance (Random Forest) — descriptive"

    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Cannot find {FEATURES_CSV} — set the path at the top of this script.")

    df = pd.read_csv(FEATURES_CSV)
    meta = [c for c in ["subject_id","label","epoch_idx"] if c in df.columns]
    feat_names = [c for c in df.columns if c not in meta]
    X = df[feat_names].to_numpy(dtype=float)
    y = (df["label"].astype(str).str.upper() == "ADHD").astype(int).to_numpy()
    print(f"Loaded {X.shape[0]} epochs x {X.shape[1]} features; ADHD={y.sum()}, Control={(y==0).sum()}")

    Xs = StandardScaler().fit_transform(X)
    mdl.fit(Xs, y)
    imp = mdl.feature_importances_
    order = np.argsort(imp)[::-1][:TOP][::-1]
    labels = [relabel(feat_names[i]) for i in order]

    print("Top features (relabelled):")
    for i in order[::-1]:
        print(f"  {relabel(feat_names[i]):28s}  {imp[i]:.5f}")

    fig, ax = plt.subplots(figsize=(7.8, 6.5), dpi=300)
    ax.barh(range(len(order)), imp[order], color=TEAL)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Importance (gain)"); ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUTDIR, "14_feature_importance.png")
    pdf = os.path.join(OUTDIR, "14_feature_importance.pdf")
    fig.savefig(png, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved:\n  {os.path.abspath(png)}\n  {os.path.abspath(pdf)}")

if __name__ == "__main__":
    main()