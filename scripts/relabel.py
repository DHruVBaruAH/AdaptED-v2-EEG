"""
relabel_feature_importance.py
--------------------------------------------------------------------
Drop-in replacement for fig_feature_importance() that prints REAL
10-20 channel names and band names instead of ch0-18 / band0-4.

Mapping (from your CHANNELS constant and FREQ_BANDS):
  ch0..ch18 -> Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz
  band0..4  -> delta, theta, alpha, beta, gamma

Other feature stems are tidied for display:
  pow_freq_bands_chN_bandB -> "<CH> <band> power"
  app_entropy_chN          -> "<CH> approx. entropy"
  std_chN                  -> "<CH> std"
  ptp_amp_chN              -> "<CH> peak-to-peak"
  kurtosis_chN             -> "<CH> kurtosis"
  skewness_chN             -> "<CH> skewness"
  katz_fd_chN              -> "<CH> Katz FD"
  hjorth_mobility_chN      -> "<CH> Hjorth mob."
  hjorth_complexity_chN    -> "<CH> Hjorth comp."

Honest-scope note: importance is DESCRIPTIVE (XGBoost gain on all data) -
association, not causation. Caption it that way.

Usage: replace your fig_feature_importance() with the version below, or
import relabel() and call it on your existing feature-name list.
"""

import re

CHANNELS = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
            "F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"]
BANDS = ["delta","theta","alpha","beta","gamma"]

_STEM = {
    "app_entropy": "approx. entropy",
    "std": "std",
    "ptp_amp": "peak-to-peak",
    "kurtosis": "kurtosis",
    "skewness": "skewness",
    "katz_fd": "Katz FD",
    "hjorth_mobility": "Hjorth mob.",
    "hjorth_complexity": "Hjorth comp.",
    "mean": "mean",
}

def _ch(n):  return CHANNELS[int(n)] if 0 <= int(n) < len(CHANNELS) else f"ch{n}"
def _band(b): return BANDS[int(b)] if 0 <= int(b) < len(BANDS) else f"band{b}"

def relabel(name):
    """Translate one raw feature name into a human-readable 10-20 label."""
    m = re.match(r"pow_freq_bands_ch(\d+)_band(\d+)", name)
    if m:
        return f"{_ch(m.group(1))} {_band(m.group(2))} power"
    m = re.match(r"([a-z_]+?)_ch(\d+)$", name)
    if m:
        stem, n = m.group(1), m.group(2)
        return f"{_ch(n)} {_STEM.get(stem, stem)}"
    return name  # leave anything unrecognised untouched

# ---------------------------------------------------------------
# Drop-in figure function (matches your make_figures.py style)
# ---------------------------------------------------------------
def fig_feature_importance(X, y, feat_names, top=20,
                           OUTDIR="figures", TEAL="#0F6E56", save=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    try:
        from xgboost import XGBClassifier
        mdl = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            eval_metric="logloss", random_state=42, verbosity=0)
        title = "Feature importance (XGBoost gain) — descriptive"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        mdl = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        title = "Feature importance (Random Forest) — descriptive"

    Xs = StandardScaler().fit_transform(X)
    mdl.fit(Xs, y)
    imp = mdl.feature_importances_
    order = np.argsort(imp)[::-1][:top][::-1]
    labels = [relabel(feat_names[i]) for i in order]

    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    ax.barh(range(len(order)), imp[order], color=TEAL)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    if save:
        save(fig, "14_feature_importance")
    else:
        for ext in ("png", "pdf"):
            fig.savefig(f"{OUTDIR}/14_feature_importance.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return [feat_names[i] for i in order], labels

if __name__ == "__main__":
    # quick self-test of the mapping on the names visible in your figure
    tests = ["pow_freq_bands_ch18_band4","pow_freq_bands_ch17_band4","app_entropy_ch11",
             "std_ch1","pow_freq_bands_ch2_band3","ptp_amp_ch5","hjorth_mobility_ch9",
             "kurtosis_ch12","katz_fd_ch12"]
    for t in tests:
        print(f"{t:32s} -> {relabel(t)}")
