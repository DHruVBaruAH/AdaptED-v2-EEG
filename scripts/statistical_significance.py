import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations

warnings.filterwarnings("ignore")

PREDICTIONS_PATH = "results/loso_predictions.csv"


def main():
    print("Loading per-subject predictions...")
    df = pd.read_csv(PREDICTIONS_PATH)
    print(f"  Subjects: {df['subject_id'].nunique()}")
    print(f"  Algorithms: {df['algorithm'].unique().tolist()}\n")

    # Pivot to get one row per subject, one column per algorithm
    prob_pivot = df.pivot(
        index="subject_id",
        columns="algorithm",
        values="mean_prob_adhd"
    )
    pred_pivot = df.pivot(
        index="subject_id",
        columns="algorithm",
        values="predicted_label"
    )
    true_labels = df.drop_duplicates("subject_id").set_index("subject_id")["true_label"]

    algorithms = prob_pivot.columns.tolist()
    print(f"Algorithms found: {algorithms}\n")

    # ----------------------------------------------------------------
    # Per-subject accuracy (correct=1, wrong=0) for each algorithm
    # ----------------------------------------------------------------
    correct = pd.DataFrame(index=prob_pivot.index)
    for alg in algorithms:
        correct[alg] = (pred_pivot[alg] == true_labels).astype(int)

    print("Per-algorithm accuracy (subject level):")
    for alg in algorithms:
        acc = correct[alg].mean()
        print(f"  {alg:<22}: {acc:.4f} ({correct[alg].sum()}/{len(correct)})")

    # ----------------------------------------------------------------
    # Wilcoxon signed-rank test — pairwise between all algorithms
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Wilcoxon Signed-Rank Test (pairwise)")
    print("H0: no difference in per-subject accuracy between algorithms")
    print(f"{'='*70}")
    print(f"\n{'Comparison':<40} {'W-stat':>10} {'p-value':>12} {'Significant':>14}")
    print("-" * 78)

    alpha = 0.05
    for alg_a, alg_b in combinations(algorithms, 2):
        diff = correct[alg_a].values - correct[alg_b].values
        # Wilcoxon requires non-zero differences
        if np.all(diff == 0):
            print(f"{alg_a} vs {alg_b:<20} {'N/A':>10} {'N/A':>12} {'No difference':>14}")
            continue
        try:
            stat, p = wilcoxon(correct[alg_a].values, correct[alg_b].values,
                               alternative="two-sided")
            sig = "YES ***" if p < alpha else "no"
            print(f"{alg_a+' vs '+alg_b:<40} {stat:>10.1f} {p:>12.4f} {sig:>14}")
        except Exception as e:
            print(f"{alg_a} vs {alg_b}: ERROR -> {e}")

    # ----------------------------------------------------------------
    # Binomial test — is best model better than chance?
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Binomial Test — Is each algorithm better than chance (50%)?")
    print(f"{'='*70}")
    print(f"\n{'Algorithm':<22} {'Correct':>8} {'Total':>8} {'p-value':>12} {'Better than chance':>20}")
    print("-" * 75)

    for alg in algorithms:
        n_correct = correct[alg].sum()
        n_total = len(correct)
        # scipy >= 1.7 uses binomtest
        try:
            from scipy.stats import binomtest
            result = binomtest(n_correct, n_total, p=0.5, alternative="greater")
            p = result.pvalue
        except ImportError:
            from scipy.stats import binom_test
            p = binom_test(n_correct, n_total, p=0.5, alternative="greater")
        sig = "YES ***" if p < alpha else "no"
        print(f"{alg:<22} {n_correct:>8} {n_total:>8} {p:>12.4f} {sig:>20}")

    # ----------------------------------------------------------------
    # AUC-based comparison using per-subject probabilities
    # ----------------------------------------------------------------
    from sklearn.metrics import roc_auc_score
    y_true = (true_labels == "ADHD").astype(int)

    print(f"\n{'='*70}")
    print("Subject-Level AUC Summary")
    print(f"{'='*70}")
    print(f"\n{'Algorithm':<22} {'AUC':>10}")
    print("-" * 35)
    for alg in algorithms:
        auc = roc_auc_score(y_true, prob_pivot[alg])
        print(f"{alg:<22} {auc:>10.4f}")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    best_alg = correct.mean().idxmax()
    best_acc = correct.mean().max()
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Best algorithm by subject accuracy: {best_alg} ({best_acc:.4f})")
    print(f"Statistical significance threshold: alpha = {alpha}")
    print(f"Note: Wilcoxon test on per-subject correct/incorrect (binary).")
    print(f"      Binomial test checks if each algorithm beats random chance.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()