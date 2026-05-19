from sklearn.svm import SVC
from scipy.sparse.linalg import _special_sparse_arrays
from scipy.sparse.linalg import _special_sparse_arrays
from scipy.sparse.linalg import _special_sparse_arrays
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURES_PATH = "data/features/eeg_features.csv"
RESULTS_DIR = "results"

ALGORITHMS = {
    "XGBoost": XGBClassifier(
        n_estimators=400, 
        max_depth=4, 
        learning_rate=0.1,
        subsample=0.8, 
        colsample_bytree=1.0,
        objective="binary:logistic", 
        random_state=42,
        n_jobs=-1, verbosity=0, 
        eval_metric="logloss",
        use_label_encoder=False,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),
    "LogisticRegression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        solver="lbfgs",
    ),
    "SVM": SVC(
        kernel="rbf",
        C=10.0,
        gamma=0.001,
        probability=True,
        random_state=42,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def aggregate_to_subject(epoch_preds, groups):
    """
    Aggregate per-epoch probabilities to subject-level prediction.
    Returns: dict {subject_id: mean_prob}
    """
    df = pd.DataFrame({"subject": groups, "prob": epoch_preds})
    return df.groupby("subject")["prob"].mean().to_dict()


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Returns dict of AUC, acc, F1, precision, recall."""
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load features ----
    print(f"Loading features from {FEATURES_PATH} ...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Subjects: {df['subject_id'].nunique()}")
    print(f"  Epoch class distribution: {dict(df['label'].value_counts())}\n")

    # ---- Prepare X, y, groups ----
    X = df.drop(columns=["subject_id", "label", "epoch_idx"]).to_numpy(dtype=np.float64)
    y = (df["label"] == "ADHD").astype(int).to_numpy()  # ADHD=1, Control=0
    groups = df["subject_id"].to_numpy()

    # Subject-level labels for subject-level metric computation
    subj_labels = df.groupby("subject_id")["label"].first()
    subj_labels = (subj_labels == "ADHD").astype(int).to_dict()

    # ---- LOSO setup ----
    logo = LeaveOneGroupOut()
    n_folds = len(np.unique(groups))
    print(f"Running LOSO with {n_folds} folds (one per subject)...\n")

    # Storage
    results = {alg: {"epoch": [], "subject": [], "subject_truth": []} for alg in ALGORITHMS}
    per_subject_preds = []  # for saving full per-subject output

    for alg_name, model_template in ALGORITHMS.items():
        print(f"{'='*70}")
        print(f"Algorithm: {alg_name}")
        print(f"{'='*70}")

        all_epoch_y_true = []
        all_epoch_y_prob = []
        all_epoch_groups = []

        t_start = time.time()
        for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subject = groups[test_idx][0]

            # Scale features (fit on train only, no leakage)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Clone the model template for this fold
            # (need a fresh instance per fold to avoid stale state)
            if alg_name == "XGBoost":
                model = XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    objective="binary:logistic", random_state=42,
                    n_jobs=-1, verbosity=0, eval_metric="logloss",
                    use_label_encoder=False,
                )
            elif alg_name == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
                )
            elif alg_name == "SVM":
                model = SVC(
                kernel="rbf",
                C=10.0,
                gamma=0.001,
                probability=True,
                random_state=42,
            )
            else:  # LogisticRegression
                model = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=42, n_jobs=-1, solver="lbfgs",
                )

            # Train and predict
            model.fit(X_train_s, y_train)
            test_probs = model.predict_proba(X_test_s)[:, 1]

            # Store per-epoch results
            all_epoch_y_true.extend(y_test.tolist())
            all_epoch_y_prob.extend(test_probs.tolist())
            all_epoch_groups.extend([test_subject] * len(test_idx))

            if fold_i % 20 == 0 or fold_i == n_folds:
                elapsed = time.time() - t_start
                print(f"  Fold {fold_i:3d}/{n_folds}: subject={test_subject}, "
                      f"test epochs={len(test_idx)}, elapsed={elapsed:.1f}s")

        # ---- Epoch-level metrics ----
        epoch_metrics = compute_metrics(all_epoch_y_true, all_epoch_y_prob)
        results[alg_name]["epoch"] = epoch_metrics

        # ---- Subject-level aggregation ----
        subj_probs = aggregate_to_subject(all_epoch_y_prob, all_epoch_groups)
        subj_y_true = [subj_labels[s] for s in subj_probs.keys()]
        subj_y_prob = [subj_probs[s] for s in subj_probs.keys()]
        subj_metrics = compute_metrics(subj_y_true, subj_y_prob)
        results[alg_name]["subject"] = subj_metrics

        # Save per-subject predictions
        for s, p in subj_probs.items():
            per_subject_preds.append({
                "subject_id": s,
                "true_label": "ADHD" if subj_labels[s] == 1 else "Control",
                "algorithm": alg_name,
                "mean_prob_adhd": p,
                "predicted_label": "ADHD" if p >= 0.5 else "Control",
            })

        print(f"\n[{alg_name}] Epoch-level metrics:")
        for k, v in epoch_metrics.items():
            print(f"    {k}: {v:.4f}")
        print(f"[{alg_name}] Subject-level metrics:")
        for k, v in subj_metrics.items():
            print(f"    {k}: {v:.4f}")
        print()

    # ---- Save per-subject predictions ----
    preds_df = pd.DataFrame(per_subject_preds)
    preds_path = os.path.join(RESULTS_DIR, "loso_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Per-subject predictions saved to: {preds_path}")

    # ---- Save summary text ----
    summary_path = os.path.join(RESULTS_DIR, "loso_summary.txt")
    with open(summary_path, "w") as f:
        f.write("LOSO Cross-Validation Results\n")
        f.write("=" * 70 + "\n\n")
        for alg_name in ALGORITHMS:
            f.write(f"\n{alg_name}\n{'-'*40}\n")
            f.write(f"  Epoch-level metrics ({len(df)} epochs):\n")
            for k, v in results[alg_name]["epoch"].items():
                f.write(f"    {k}: {v:.4f}\n")
            f.write(f"  Subject-level metrics ({df['subject_id'].nunique()} subjects):\n")
            for k, v in results[alg_name]["subject"].items():
                f.write(f"    {k}: {v:.4f}\n")
    print(f"Summary text saved to: {summary_path}")

    # ---- Bar chart comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metric_names = ["AUC", "Accuracy", "F1", "Precision", "Recall"]
    algs = list(ALGORITHMS.keys())
    x = np.arange(len(metric_names))
    width = 0.25

    for level_i, level in enumerate(["epoch", "subject"]):
        ax = axes[level_i]
        for i, alg in enumerate(algs):
            vals = [results[alg][level][m] for m in metric_names]
            ax.bar(x + i*width, vals, width, label=alg)
        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title(f"{level.capitalize()}-level metrics")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Score")

    plt.suptitle("LOSO Cross-Validation Results (AdaptED EEG Model)", fontsize=14)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "loso_summary.png")
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"Comparison chart saved to: {chart_path}")

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Algorithm':<22} {'Epoch AUC':>12} {'Subject AUC':>14} {'Subject Acc':>14}")
    print("-" * 70)
    for alg_name in ALGORITHMS:
        e_auc = results[alg_name]["epoch"]["AUC"]
        s_auc = results[alg_name]["subject"]["AUC"]
        s_acc = results[alg_name]["subject"]["Accuracy"]
        print(f"{alg_name:<22} {e_auc:>12.4f} {s_auc:>14.4f} {s_acc:>14.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()