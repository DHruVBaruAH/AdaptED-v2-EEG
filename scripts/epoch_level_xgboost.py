import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURES_PATH = "data/features/eeg_features.csv"

def main():
    print("Loading features...")
    df = pd.read_csv(FEATURES_PATH)

    feature_cols = [c for c in df.columns
                   if c not in ["subject_id", "label", "epoch_idx"]]

    # EPOCH LEVEL — replicating Garcia-Ponsoda's approach
    # Epochs split randomly, same subject appears in train and test
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = (df["label"] == "ADHD").astype(int).to_numpy()

    print(f"Epoch-level dataset: {X.shape[0]} epochs, {X.shape[1]} features")
    print(f"Algorithm: XGBoost — replicating Garcia-Ponsoda setup")
    

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_y_true = []
    all_y_prob = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            objective="binary:logistic", random_state=42,
            n_jobs=-1, verbosity=0, eval_metric="logloss",
            use_label_encoder=False,
        )

        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)

        fold_acc = accuracy_score(y_test, preds)
        print(f"  Fold {fold_i}: test epochs={len(test_idx)}, "
              f"acc={fold_acc:.4f}")

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(probs.tolist())

    auc = roc_auc_score(all_y_true, all_y_prob)
    acc = accuracy_score(all_y_true,
                        (np.array(all_y_prob) >= 0.5).astype(int))
    f1 = f1_score(all_y_true,
                 (np.array(all_y_prob) >= 0.5).astype(int))

    print(f"\n{'='*65}")
    print(f"XGBOOST — EPOCH-LEVEL 5-FOLD (Replicating Garcia-Ponsoda)")
    print(f"{'='*65}")
    print(f"AUC:      {auc:.4f}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1:       {f1:.4f}")
    print(f"\n{'='*65}")
    print(f"DIRECT COMPARISON")
    print(f"{'='*65}")
    print(f"Garcia-Ponsoda XGBoost epoch-level 5-fold:  ~86.00%")
    print(f"Our XGBoost epoch-level 5-fold:      {acc*100:.2f}%")
    print(f"Our SVM epoch-level 5-fold:          93.28%")
    print(f"Our SVM LOSO:           74.17%")
    print(f"\nConclusion:")
    print(f"Under identical leaky evaluation as Garcia-Ponsoda,")
    print(f"our pipeline with XGBoost scores {acc*100:.2f}% — directly")
    print(f"comparable to their 86%.")
    print(f"Under honest LOSO evaluation our SVM scores 74.17%")
    print(f"— the clinically meaningful number.")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()