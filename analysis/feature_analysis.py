import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_dir",
        required=True
    )

    parser.add_argument(
        "--tsne_samples",
        type=int,
        default=3000,
        help="Maximum number of samples for t-SNE."
    )

    args = parser.parse_args()

    features = np.load(
        os.path.join(
            args.result_dir,
            "features.npy"
        )
    )

    labels = np.load(
        os.path.join(
            args.result_dir,
            "labels.npy"
        )
    )

    predictions = np.load(
        os.path.join(
            args.result_dir,
            "predictions.npy"
        )
    )

    print()

    print("Feature Shape :", features.shape)

    print("Labels :", labels.shape)

    print("Predictions :", predictions.shape)

    assert len(features) == len(labels)

    assert len(labels) == len(predictions)

    analysis_dir = os.path.join(
        args.result_dir,
        "analysis"
    )

    os.makedirs(
        analysis_dir,
        exist_ok=True
    )
    stats = {

        "num_samples": int(features.shape[0]),

        "feature_dimension": int(features.shape[1]),

        "mean": float(features.mean()),

        "std": float(features.std()),

        "min": float(features.min()),

        "max": float(features.max())
    }
    with open(
        os.path.join(
            analysis_dir,
            "feature_statistics.json"
        ),
        "w"
    ) as f:

        json.dump(
            stats,
            f,
            indent=4
        )

    print("\nRunning PCA...")
    
    pca = PCA(
        n_components=2,
        random_state=42
    )

    features_pca = pca.fit_transform(features)
    plt.figure(figsize=(8,8))

    scatter = plt.scatter(

        features_pca[:,0],

        features_pca[:,1],

        c=labels,

        s=8,

        alpha=0.6

    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title("PCA of Student Features")

    plt.colorbar(scatter)

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            analysis_dir,
            "pca_2d.png"
        ),

        dpi=300
    )

    plt.close()

    print("\ PCA finished...")
    print("\nRunning TSNE...")

    MAX_TSNE_SAMPLES = args.tsne_samples

    if len(features) > MAX_TSNE_SAMPLES:

        _, sample_idx = train_test_split(
            np.arange(len(features)),
            test_size=MAX_TSNE_SAMPLES,
            stratify=labels,
            random_state=42
        )

        tsne_features = features[sample_idx]
        tsne_labels = labels[sample_idx]

    else:

        tsne_features = features
        tsne_labels = labels

    
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    features_tsne = tsne.fit_transform(
        tsne_features
    )

    plt.figure(figsize=(8,8))

    scatter = plt.scatter(
        features_tsne[:,0],
        features_tsne[:,1],
        c=tsne_labels,
        s=8,
        alpha=0.7
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE of Student Feature Space")

    plt.colorbar(scatter)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            analysis_dir,
            "tsne_2d.png"
        ),
        dpi=300
    )
    print("✅ PCA completed.")
    print("📂 Saved: pca_2d.png")

    print("\nRunning t-SNE on", len(tsne_features), "samples...")
    print("⏳ This may take several minutes.")


    plt.savefig(
        os.path.join(
            analysis_dir,
            "tsne_2d.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()

    variance = {

        "pc1": float(
            pca.explained_variance_ratio_[0]
        ),

        "pc2": float(
            pca.explained_variance_ratio_[1]
        ),

        "total": float(
            np.sum(
                pca.explained_variance_ratio_
            )
        )
    }
    with open(
        os.path.join(
            analysis_dir,
            "explained_variance.json"
        ),
        "w"
    ) as f:

        json.dump(
            variance,
            f,
            indent=4
        )
    plt.figure(figsize=(6,4))

    plt.bar(
        ["PC1","PC2"],
        pca.explained_variance_ratio_
    )

    plt.ylabel("Explained Variance Ratio")

    plt.title("Principal Component Variance")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            analysis_dir,
            "pca_variance.png"
        ),
        dpi=300
    )
    print("\nFeature analysis completed successfully.")
    print(f"Results saved to: {analysis_dir}")

    plt.close()

if __name__ == "__main__":
    main()