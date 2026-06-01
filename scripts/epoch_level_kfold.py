import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

FEATURES_PATH = "data/features/eeg_features.csv"

def main():
    print("Loading features...")
    df = pd.read_csv(FEATURES_PATH)

    feature_cols = [c for c in df.columns
                   if c not in ["subject_id", "label", "epoch_idx"]]

    # EPOCH LEVEL — no subject grouping, direct random split
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = (df["label"] == "ADHD").astype(int).to_numpy()

    print(f"Epoch-level dataset: {X.shape[0]} epochs, {X.shape[1]} features")
    print(f"WARNING: This evaluation has subject leakage — same subject")
    print(f"epochs appear in both train and test folds.\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_y_true = []
    all_y_prob = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = SVC(
            kernel="rbf", C=10, gamma=0.001,
            probability=True, random_state=42,
        )
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(probs.tolist())

        pred = (np.array(probs) >= 0.5).astype(int)
        fold_acc = accuracy_score(y_test, pred)
        print(f"  Fold {fold_i}: test epochs={len(test_idx)}, acc={fold_acc:.4f}")

    auc = roc_auc_score(all_y_true, all_y_prob)
    acc = accuracy_score(all_y_true,
                        (np.array(all_y_prob) >= 0.5).astype(int))

    print(f"\n{'='*60}")
    print(f"EPOCH-LEVEL 5-FOLD CV RESULTS (SVM)")
    print(f"{'='*60}")
    print(f"AUC:      {auc:.4f}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nComparison:")
    print(f"  Epoch-level 5-fold (leaky):    AUC {auc:.4f}, Acc {acc*100:.2f}%")
    print(f"  Subject-level LOSO (honest):   AUC 0.7813, Acc 74.17%")
    print(f"\nThe gap above = subject leakage inflation")
    print(f"comparable to the leaky number above, not our LOSO result.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()