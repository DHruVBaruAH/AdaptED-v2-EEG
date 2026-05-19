# AdaptED v2 — EEG-Based ADHD Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![MNE](https://img.shields.io/badge/MNE-EEG-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A rigorous, reproducible ML pipeline for ADHD classification from EEG signals using Leave-One-Subject-Out (LOSO) cross-validation.**

[Overview](#overview) • [Results](#results) • [Pipeline](#pipeline) • [Setup](#setup) • [Usage](#usage) • [Methodology](#methodology) • [Citation](#citation)

</div>

---

## Overview

AdaptED v2 builds a complete EEG-based ADHD classification system on the **Nasrabadi dataset** — the same benchmark used by Esas (2023) and García-Ponsoda et al. (2024).

**What makes this different from prior work:**

| | Esas 2023 | García-Ponsoda 2024 | **AdaptED v2 (Ours)** |
|---|---|---|---|
| Validation | k-fold CV | 5-fold CV | **LOSO (gold standard)** |
| Channels | 19 | 3 only | **All 19** |
| Algorithms compared | 1 | 1 | **4 (systematic)** |
| Subject leakage | Yes | Yes | **No** |
| Statistical testing | No | No | **Yes** |

---

## Results

### Subject-Level Performance (LOSO)

| Algorithm | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **SVM** ⭐ | **0.7813** | **75.00%** | **0.7941** | **0.7297** | **0.8689** |
| Logistic Regression | 0.7683 | 71.67% | 0.7188 | 0.6866 | 0.7541 |
| XGBoost | 0.7508 | 74.17% | 0.7634 | 0.7143 | 0.8197 |
| Random Forest | 0.7146 | 69.17% | 0.7413 | 0.6463 | 0.8689 |

### Validation Gap Analysis

> Same model. Same data. Different evaluation method. Different number.
> This proves reported accuracy is a property of the evaluation — not just the model.

| Algorithm | 5-Fold AUC | 5-Fold Acc | LOSO AUC | LOSO Acc |
|---|---|---|---|---|
| XGBoost | 0.7474 | 65.83% | 0.7508 | 74.17% |
| Logistic Regression | 0.7316 | 67.50% | 0.7683 | 71.67% |
| Random Forest | 0.7265 | 65.00% | 0.7146 | 69.17% |

### Statistical Significance

All algorithms perform significantly above random chance (binomial test, **p < 0.001**).

---

## Pipeline

```
Raw EEG CSV (Nasrabadi Dataset)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │  Bandpass 0.5–40 Hz + Notch 50 Hz
│ eeg_preprocessing   │  Average reference, MNE RawArray
└────────┬────────────┘
         │  data/cleaned/*.npy
         ▼
┌─────────────────────┐
│ Segmentation +      │  5-sec epochs, 50% overlap
│ Feature Extraction  │  266 features via mne-features
└────────┬────────────┘
         │  data/features/eeg_features.csv
         ▼
┌─────────────────────┐
│  LOSO Validation    │  120 folds, one per subject
│  4 Algorithms       │  XGBoost · RF · LR · SVM
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Analysis           │  PCA · t-SNE · Wilcoxon · Binomial
└─────────────────────┘
```

---

## Project Structure

```
AdaptED_v2/
├── scripts/
│   ├── eeg_preprocessing.py          # Step 1 — bandpass + notch filtering
│   ├── eeg_segmentation_features.py  # Step 2 — epoch segmentation + feature extraction
│   ├── eeg_loso_xgboost.py           # Step 3 — LOSO validation (4 algorithms)
│   ├── kfold_comparison.py           # Step 4 — 5-fold vs LOSO comparison
│   ├── dimensionality_reduction.py   # Step 5 — PCA + t-SNE visualization
│   ├── statistical_significance.py   # Step 6 — Wilcoxon + binomial tests
│   ├── tune_xgboost.py               # Optional — XGBoost hyperparameter tuning
│   └── add_asymmetry_features.py     # Optional — hemispheric asymmetry features
├── data/
│   ├── cleaned/                      # Preprocessed .npy signals (generated)
│   │   └── _metadata.csv             # Subject metadata
│   └── features/                     # Extracted feature CSV (generated)
├── results/                          # Output plots, summaries, predictions
│   ├── loso_summary.png              # Algorithm comparison bar chart
│   ├── loso_summary.txt              # Full metrics report
│   ├── loso_predictions.csv          # Per-subject predictions
│   ├── pca_visualization.png         # PCA 2D scatter
│   ├── pca_variance_curve.png        # Explained variance curve
│   └── tsne_visualization.png        # t-SNE 2D scatter
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- Windows / Linux / macOS
- ~4GB RAM minimum

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/DHruVBaruAH/AdaptED-v2-EEG.git
cd AdaptED-v2-EEG
```

**2. Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

### Dataset

Download the Nasrabadi EEG dataset from IEEE Dataport:
> https://ieee-dataport.org/open-access/eeg-data-adhd-control-children

Place the raw CSV at:
```
D:/ADHD_personalized/eeg_data/adhdata.csv
```

Or update `RAW_CSV_PATH` in `scripts/eeg_preprocessing.py` to match your path.

---

## Usage

Run scripts **in order**. Each step depends on the previous.

### Step 1 — Preprocessing
```bash
python scripts/eeg_preprocessing.py
```
Applies bandpass (0.5–40 Hz) and notch (50 Hz) filters to all 120 subjects.
Saves cleaned signals to `data/cleaned/`.
**Runtime:** ~8 minutes

### Step 2 — Feature Extraction
```bash
python scripts/eeg_segmentation_features.py
```
Segments signals into 5-second epochs with 50% overlap.
Extracts 266 features per epoch using mne-features.
Saves to `data/features/eeg_features.csv`.
**Runtime:** ~8 minutes

### Step 3 — LOSO Validation
```bash
python scripts/eeg_loso_xgboost.py
```
Runs 120-fold LOSO with XGBoost, Random Forest, Logistic Regression, and SVM.
Saves results to `results/`.
**Runtime:** ~60–90 minutes

### Step 4 — Validation Gap Analysis (Optional)
```bash
python scripts/kfold_comparison.py
```
Compares 5-fold CV vs LOSO on the same data.
**Runtime:** ~1 minute

### Step 5 — Dimensionality Reduction (Optional)
```bash
python scripts/dimensionality_reduction.py
```
Generates PCA and t-SNE visualizations of subject-level features.
**Runtime:** ~3 minutes

### Step 6 — Statistical Significance (Optional)
```bash
python scripts/statistical_significance.py
```
Runs Wilcoxon signed-rank test and binomial test on per-subject predictions.
**Runtime:** ~5 seconds

---

## Methodology

### Preprocessing
- Bandpass FIR filter: 0.5–40 Hz
- Notch filter: 50 Hz (power line interference)
- Average EEG reference
- Minimum recording length: 9,250 samples (~72 seconds, matching Esas 2023)

### Feature Extraction
266 features extracted per epoch across 19 channels using `mne-features`:

| Feature | Description |
|---|---|
| Mean, Std, Kurtosis, Skewness | Statistical moments |
| Peak-to-peak amplitude | Signal range |
| Band powers | Delta, Theta, Alpha, Beta, Gamma |
| Approximate entropy | Signal complexity |
| Katz fractal dimension | Non-linear dynamics |
| Hjorth mobility & complexity | Signal regularity |

### Validation
- **LOSO (Leave-One-Subject-Out):** 120 folds, one subject held out per fold
- Subject-level prediction by averaging per-epoch probabilities
- No data leakage between subjects guaranteed

### Why LOSO over k-fold
In k-fold CV on EEG data, epochs from the same subject can appear in both training and test folds — the model sees familiar brain patterns during testing. LOSO guarantees every test subject is completely unseen during training, simulating real clinical deployment where every patient is new.

---

## Visualizations

<table>
<tr>
<td><b>t-SNE — Subject Feature Space</b></td>
<td><b>PCA Explained Variance</b></td>
</tr>
<tr>
<td>Shows inter-subject variability — ADHD and Control overlap confirms the genuine difficulty of the classification task.</td>
<td>33 components explain 90% of variance across 266 features, justifying high-dimensional classification methods like SVM.</td>
</tr>
</table>

---

## Key Findings

- **SVM with RBF kernel** achieves the best subject-level AUC (**0.7774**) and accuracy (**75%**) under LOSO
- **Logistic Regression** is competitive (AUC 0.7683), suggesting features have strong linear separability
- **Validation method drives reported accuracy** — same model shows different numbers under k-fold vs LOSO
- All algorithms significantly outperform random chance (**p < 0.001**, binomial test)
- No statistically significant difference between algorithms (Wilcoxon, p > 0.05), consistent with n=120 sample size limitations

---

## Limitations

- No ICA artifact removal — eye blink and muscle artifacts may remain in signal
- Dataset limited to 120 subjects — insufficient power for strong pairwise significance tests
- Binary classification only — ADHD subtypes not distinguished
- Features computed on 5-second epochs — short-term dynamics only

---

## Citation

If you use this code or build on this work, please cite the Nasrabadi dataset:

```bibtex
@data{nasrabadi2020,
  author    = {Nasrabadi, Ali Motie},
  title     = {EEG Data for ADHD / Control Children},
  year      = {2020},
  publisher = {IEEE Dataport},
  doi       = {10.21227/rzfh-zn36},
  url       = {https://ieee-dataport.org/open-access/eeg-data-adhd-control-children}
}
```

---

## References

- Esas, M. Y. (2023). EEG-based ADHD detection using machine learning.
- García-Ponsoda, S. et al. (2024). EEG classification with fractal and wavelet features.
- Kim, J. et al. (2025). Subject-independent EEG classification for ADHD. *(LOSO benchmark)*
- Nasrabadi, A. M. et al. (2020). EEG data for ADHD/Control children. IEEE Dataport.

---

<div align="center">
Made with Python · MNE · scikit-learn · XGBoost
</div>
