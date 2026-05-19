import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

FEATURES_PATH = "data/features/eeg_features.csv"
RESULTS_DIR = "results"

def main():
    print("Loading features...")
    df = pd.read_csv(FEATURES_PATH)

    # Use subject-level mean features — one row per subject
    feature_cols = [c for c in df.columns 
                   if c not in ["subject_id", "label", "epoch_idx"]]
    
    subj_features = df.groupby("subject_id")[feature_cols].mean().reset_index()
    subj_labels = df.groupby("subject_id")["label"].first().reset_index()
    subj_data = subj_features.merge(subj_labels, on="subject_id")

    X = subj_data[feature_cols].to_numpy(dtype=np.float64)
    y = subj_data["label"].to_numpy()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset: {X.shape[0]} subjects, {X.shape[1]} features")

    # ----------------------------------------------------------------
    # PCA
    # ----------------------------------------------------------------
    print("\nRunning PCA...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.argmax(explained >= 0.90) + 1
    n_95 = np.argmax(explained >= 0.95) + 1
    print(f"  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")
    print(f"  Top 2 components explain: {explained[1]*100:.1f}% variance")

    # PCA 2D scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color, marker in [("ADHD", "#e74c3c", "o"), 
                                   ("Control", "#2ecc71", "s")]:
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=color, marker=marker, alpha=0.7,
                  label=f"{label} (n={mask.sum()})", s=60)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA — Subject-Level EEG Features (Nasrabadi Dataset)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pca_path = f"{RESULTS_DIR}/pca_visualization.png"
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f"  PCA plot saved to: {pca_path}")

    # Explained variance curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 51), explained * 100, color="#3498db", linewidth=2)
    ax.axhline(90, color="#e74c3c", linestyle="--", alpha=0.7, label="90% threshold")
    ax.axhline(95, color="#e67e22", linestyle="--", alpha=0.7, label="95% threshold")
    ax.axvline(n_90, color="#e74c3c", linestyle=":", alpha=0.5)
    ax.axvline(n_95, color="#e67e22", linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA Explained Variance — EEG Features")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    var_path = f"{RESULTS_DIR}/pca_variance_curve.png"
    plt.savefig(var_path, dpi=150)
    plt.close()
    print(f"  Variance curve saved to: {var_path}")

    # ----------------------------------------------------------------
    # t-SNE
    # ----------------------------------------------------------------
    print("\nRunning t-SNE (this takes a minute)...")
    # Use top PCA components as input to t-SNE for stability
    X_pca_50 = X_pca  # already 50 components
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    X_tsne = tsne.fit_transform(X_pca_50)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color, marker in [("ADHD", "#e74c3c", "o"),
                                   ("Control", "#2ecc71", "s")]:
        mask = y == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=color, marker=marker, alpha=0.7,
                  label=f"{label} (n={mask.sum()})", s=60)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title("t-SNE — Subject-Level EEG Features (Nasrabadi Dataset)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    tsne_path = f"{RESULTS_DIR}/tsne_visualization.png"
    plt.savefig(tsne_path, dpi=150)
    plt.close()
    print(f"  t-SNE plot saved to: {tsne_path}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"DIMENSIONALITY REDUCTION SUMMARY")
    print(f"{'='*70}")
    print(f"Original features:          {X.shape[1]}")
    print(f"Components for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")
    print(f"Variance reduction ratio:    {X.shape[1]/n_90:.1f}x (90% threshold)")
    print(f"\nOutputs saved to results/:")
    print(f"  pca_visualization.png")
    print(f"  pca_variance_curve.png")
    print(f"  tsne_visualization.png")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()