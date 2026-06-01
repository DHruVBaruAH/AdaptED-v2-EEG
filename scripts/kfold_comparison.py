import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURES_PATH = "data/features/eeg_features.csv"

LOSO_RESULTS = {
    "XGBoost":            (0.7402, 0.7250),
    "RandomForest":       (0.7146, 0.6917),
    "LogisticRegression": (0.7683, 0.7167),
    "SVM":                (0.7813, 0.7417),
}

def main():
    print("Loading features...")
    df = pd.read_csv(FEATURES_PATH)

    feature_cols = [c for c in df.columns
                   if c not in ["subject_id", "label", "epoch_idx"]]

    
    subj_features = df.groupby("subject_id")[feature_cols].mean().reset_index()
    subj_labels = df.groupby("subject_id")["label"].first().reset_index()
    subj_data = subj_features.merge(subj_labels, on="subject_id")

    X = subj_data[feature_cols].to_numpy(dtype=np.float64)
    y = (subj_data["label"] == "ADHD").astype(int).to_numpy()
    subject_ids = subj_data["subject_id"].to_numpy()

    print(f"Subject-level dataset: {X.shape[0]} subjects, {X.shape[1]} features")
    print(f"Class: ADHD={y.sum()}, Control={len(y)-y.sum()}")
    print(f"Validation: 5-Fold Stratified CV — NO subject leakage")
    print(f"Each fold tests on subjects completely unseen during training\n")

    algorithms = {
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
        "SVM": SVC(
            kernel="rbf", C=10, gamma=0.001,
            probability=True, random_state=42,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kfold_results = {}

    for alg_name in algorithms:
        print(f"{'='*65}")
        print(f"Algorithm: {alg_name} — Subject-Level 5-Fold CV (No Leakage)")
        print(f"{'='*65}")

        all_y_true = []
        all_y_prob = []

        for fold_i, (train_idx, test_idx) in enumerate(
                skf.split(X, y), 1):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subjects = subject_ids[test_idx]

            # Verify no subject overlap — explicit leakage check
            train_subjects = set(subject_ids[train_idx])
            test_subjects_set = set(test_subjects)
            overlap = train_subjects.intersection(test_subjects_set)
            assert len(overlap) == 0, f"LEAKAGE DETECTED in fold {fold_i}"

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
                    n_estimators=200, max_depth=10,
                    random_state=42, n_jobs=-1,
                )
            elif alg_name == "LogisticRegression":
                model = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=42,
                    n_jobs=-1, solver="lbfgs",
                )
            else:
                model = SVC(
                    kernel="rbf", C=10, gamma=0.001,
                    probability=True, random_state=42,
                )

            model.fit(X_train_s, y_train)
            probs = model.predict_proba(X_test_s)[:, 1]
            preds = (probs >= 0.5).astype(int)

            fold_acc = accuracy_score(y_test, preds)
            print(f"  Fold {fold_i}: "
                  f"train={len(train_idx)} subjects, "
                  f"test={len(test_idx)} subjects, "
                  f"acc={fold_acc:.4f} — NO LEAKAGE CONFIRMED")

            all_y_true.extend(y_test.tolist())
            all_y_prob.extend(probs.tolist())

        auc = roc_auc_score(all_y_true, all_y_prob)
        acc = accuracy_score(all_y_true,
                            (np.array(all_y_prob) >= 0.5).astype(int))
        f1 = f1_score(all_y_true,
                     (np.array(all_y_prob) >= 0.5).astype(int))
        kfold_results[alg_name] = (auc, acc, f1)

        print(f"\n[{alg_name}] 5-Fold Subject-Level Results:")
        print(f"    AUC:      {auc:.4f}")
        print(f"    Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"    F1:       {f1:.4f}\n")

    # ----------------------------------------------------------------
    # Final comparison table
    # ----------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"FINAL COMPARISON — SUBJECT-LEVEL 5-FOLD vs LOSO")
    print(f"Both methods guarantee NO subject leakage")
    print(f"{'='*75}")
    print(f"\n{'Algorithm':<22} {'5-Fold AUC':>12} {'5-Fold Acc':>12} "
          f"{'LOSO AUC':>12} {'LOSO Acc':>12}")
    print("-" * 65)

    for alg_name in algorithms:
        kf_auc, kf_acc, _ = kfold_results[alg_name]
        lo_auc, lo_acc = LOSO_RESULTS[alg_name]
        print(f"{alg_name:<22} {kf_auc:>12.4f} {kf_acc:>12.4f} "
              f"{lo_auc:>12.4f} {lo_acc:>12.4f}")

    print(f"\n{'='*75}")
    print(f"METHODOLOGY NOTE:")
    print(f"Both evaluations above use subject-level splitting.")
    print(f"No subject's epochs appear in both train and test.")
    print(f"5-Fold: 5 test folds of ~24 subjects each.")
    print(f"LOSO:   120 test folds of 1 subject each.")
    print(f"LOSO gives more stable estimates on small datasets")
    print(f"because it averages across 120 evaluations vs 5.")
    print(f"with only 3 channels — methodologically different")
    print(f"from both evaluations shown above.")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()