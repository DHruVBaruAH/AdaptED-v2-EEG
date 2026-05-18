import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import itertools

warnings.filterwarnings("ignore")

# Load features
df = pd.read_csv("data/features/eeg_features.csv")
X = df.drop(columns=["subject_id", "label", "epoch_idx"]).to_numpy(dtype=np.float64)
y = (df["label"] == "ADHD").astype(int).to_numpy()
groups = df["subject_id"].to_numpy()

# Subject-level aggregation helper
def subject_auc(model, X, y, groups):
    logo = LeaveOneGroupOut()
    subj_labels = {}
    for sid, lbl in zip(groups, y):
        subj_labels[sid] = lbl

    all_subj_true = []
    all_subj_prob = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        test_subject = groups[test_idx][0]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        prob = model.predict_proba(X_test_s)[:, 1].mean()

        all_subj_true.append(subj_labels[test_subject])
        all_subj_prob.append(prob)

    return roc_auc_score(all_subj_true, all_subj_prob)

# Parameter grid
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

keys = list(param_grid.keys())
values = list(param_grid.values())
combinations = list(itertools.product(*values))

print(f"Total combinations to test: {len(combinations)}")
print(f"{'='*70}")

best_auc = 0
best_params = None

for i, combo in enumerate(combinations, 1):
    params = dict(zip(keys, combo))
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    auc = subject_auc(model, X, y, groups)
    print(f"[{i:3d}/{len(combinations)}] {params} -> AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_params = params

print(f"\n{'='*70}")
print(f"BEST AUC: {best_auc:.4f}")
print(f"BEST PARAMS: {best_params}")
print(f"{'='*70}")