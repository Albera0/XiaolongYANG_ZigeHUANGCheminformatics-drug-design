import numpy as np
import matplotlib.pyplot as plt
from feature_representation import Morgan_fingerprints, RDkit_fingerprints
from DataRead_and_preprocessing import load_hiv_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# Perform PCA + KMeans clustering and visualize results.
def pca_kmeans(X, n_components=20, n_clusters=8, title="Clustering after PCA"):

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_pca)

    # Evaluate
    sil_score = silhouette_score(X_pca, labels)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"Cluster distribution: {np.bincount(labels)}")
    print(f"Silhouette score: {sil_score:.3f}")

    # Visualization
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f"{title}\n(PCA components={n_components}, clusters={n_clusters})")
    plt.tight_layout()
    plt.show()

    return X_pca, labels


# compute silhouette scores for a range of K values and visualize them

def find_best_k(X_pca, k_range=(2, 10)):
    silhouettes = []
    k_values = list(range(k_range[0], k_range[1] + 1))

    print("\nTesting different K values:")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        silhouettes.append(score)
        print(f"K={k} -> silhouette score={score:.3f}")

    # Plot silhouette scores
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, silhouettes, marker='o', color='teal')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. K for K-Means')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Find the best K
    best_k = k_values[np.argmax(silhouettes)]
    print(f"\nOptimal K: {best_k} with silhouette score: {max(silhouettes):.3f}")
    return best_k


if __name__ == "__main__":
    # Load data
    dataset = load_hiv_dataset('C:\\Users\\Zoe\\Downloads\\HIV.csv')

    # Feature extraction
    fingerprints_morgan = Morgan_fingerprints(dataset.canonical_smiles)
    X_morgan = np.array(fingerprints_morgan, dtype=float)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_morgan)

    # PCA (20D for clustering)
    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio (20 components): {pca.explained_variance_ratio_.sum():.3f}")

    # Automatically find best K
    best_k = find_best_k(X_pca, k_range=(2, 10))

    # Final clustering with best K
    X_pca_final, labels_final = pca_kmeans(X_scaled, n_components=20, n_clusters=best_k, title="Final Clustering on Morgan Fingerprints")
