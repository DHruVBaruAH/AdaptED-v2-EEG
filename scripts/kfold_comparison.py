import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURES_PATH = "data/features/eeg_features.csv"
RESULTS_DIR = "results"

ALGORITHMS = {
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="binary:logistic", random_state=42,
        n_jobs=-1, verbosity=0, eval_metric="logloss",
        use_label_encoder=False,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
    ),
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, n_jobs=-1, solver="lbfgs",
    ),
}


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


def main():
    print(f"Loading features from {FEATURES_PATH} ...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Subjects: {df['subject_id'].nunique()}")
    print(f"  Epoch class distribution: {dict(df['label'].value_counts())}\n")

    # Subject-level data — one row per subject, majority vote label
    subj_df = df.groupby("subject_id")["label"].first().reset_index()
    subj_df["y"] = (subj_df["label"] == "ADHD").astype(int)

    # For each subject, get mean features across their epochs
    feature_cols = [c for c in df.columns if c not in ["subject_id", "label", "epoch_idx"]]
    subj_features = df.groupby("subject_id")[feature_cols].mean().reset_index()
    subj_data = subj_features.merge(subj_df[["subject_id", "y"]], on="subject_id")

    X = subj_data[feature_cols].to_numpy(dtype=np.float64)
    y = subj_data["y"].to_numpy()

    print(f"Subject-level dataset: {X.shape[0]} subjects, {X.shape[1]} features")
    print(f"Class distribution: ADHD={y.sum()}, Control={len(y)-y.sum()}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for alg_name, model_template in ALGORITHMS.items():
        print(f"{'='*70}")
        print(f"Algorithm: {alg_name} — 5-Fold Stratified CV")
        print(f"{'='*70}")

        all_y_true = []
        all_y_prob = []

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

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
            else:
                model = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=42, n_jobs=-1, solver="lbfgs",
                )

            model.fit(X_train_s, y_train)
            probs = model.predict_proba(X_test_s)[:, 1]

            all_y_true.extend(y_test.tolist())
            all_y_prob.extend(probs.tolist())

            print(f"  Fold {fold_i}: test subjects={len(test_idx)}, "
                  f"ADHD={y_test.sum()}, Control={len(y_test)-y_test.sum()}")

        metrics = compute_metrics(all_y_true, all_y_prob)
        results[alg_name] = metrics

        print(f"\n[{alg_name}] 5-Fold CV metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
        print()

    # Final comparison
    print(f"\n{'='*70}")
    print(f"5-FOLD CV RESULTS vs LOSO RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Algorithm':<22} {'5-Fold AUC':>12} {'5-Fold Acc':>12} {'LOSO AUC':>12} {'LOSO Acc':>12}")
    print("-" * 70)

    # Your best LOSO numbers for comparison
    loso_results = {
        "XGBoost":            (0.7508, 0.7417),
        "RandomForest":       (0.7119, 0.6917),
        "LogisticRegression": (0.7683, 0.7167),
    }

    for alg_name in ALGORITHMS:
        kfold_auc = results[alg_name]["AUC"]
        kfold_acc = results[alg_name]["Accuracy"]
        loso_auc, loso_acc = loso_results[alg_name]
        print(f"{alg_name:<22} {kfold_auc:>12.4f} {kfold_acc:>12.4f} "
              f"{loso_auc:>12.4f} {loso_acc:>12.4f}")

    print(f"{'='*70}")
    print(f"\nNote: 5-Fold CV allows subject leakage between folds.")
    print(f"LOSO guarantees complete subject independence per fold.")
    print(f"The gap between these numbers quantifies validation optimism.")


if __name__ == "__main__":
    main()