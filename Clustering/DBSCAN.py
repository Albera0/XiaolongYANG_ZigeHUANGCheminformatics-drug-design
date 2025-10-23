import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.cluster import DBSCAN
from feature_representation import Morgan_fingerprints, dataset, Molecular_Descriptors

# Note: This process I use AI to encapsulate functions for improving readi

# ============================================================================
# 1: Morgan Fingerprint + DBSCAN (For Reference - High Noise)
# ============================================================================
# This method shows high noise ratio (~97%) due to computational
# Kept it for comparison purposes.

class MorganFingerprintClustering:
    """
    Uses sampling to handle computational complexity of Tanimoto distance, otherwise computer memory explodes
    """

    def __init__(self, dataset, sample_size=2000):
        self.dataset = dataset
        self.sample_size = sample_size
        self.fingerprints_morgan = None
        self.sample_fps = None
        self.sample_idx = None
        self.distance_matrix = None
        self.sample_labels = None
        self.all_labels = None

    def prepare_data(self):
        # Generate fingerprints and sample data.
        print("=" * 70)
        print("METHOD 1: Morgan Fingerprint + DBSCAN")
        print("=" * 70)

        self.fingerprints_morgan = Morgan_fingerprints(self.dataset.canonical_smiles)
        n_total = len(self.fingerprints_morgan)

        # Sample to reduce computational load
        self.sample_idx = sorted(random.sample(range(n_total), self.sample_size))
        self.sample_fps = [self.fingerprints_morgan[i] for i in self.sample_idx]

        print(f"Total molecules: {n_total}, sampled: {self.sample_size}")
        return self

    def calculate_tanimoto_distance(self):
        # Calculate Tanimoto distance matrix
        fingerprints = [fp for fp in self.sample_fps if isinstance(fp, ExplicitBitVect)]
        n = len(fingerprints)
        print(f"Valid fingerprints: {n}")

        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            sims = BulkTanimotoSimilarity(fingerprints[i], fingerprints)
            dist_matrix[i, :] = [1 - s for s in sims]
            if i % 500 == 0:
                print(f"Processed {i}/{n}")

        self.distance_matrix = dist_matrix
        np.save('tanimoto_distance_sample.npy', dist_matrix)
        print("Saved distance matrix to tanimoto_distance_sample.npy\n")
        return self

    def optimize_dbscan(self, eps_values=[0.35, 0.4, 0.45, 0.5]):
        # Test different eps values and select the best one
        print("Testing different eps values for DBSCAN:")

        best_eps = None
        best_score = -1
        best_labels = None

        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=5, metric='precomputed')
            labels = db.fit_predict(self.distance_matrix)

            unique, counts = np.unique(labels, return_counts=True)
            print(f"eps={eps:.2f} -> {dict(zip(unique, counts))}")

            # Calculate silhouette score only when there are multiple clusters
            if len(set(labels)) > 1 and len(set(labels)) < len(self.distance_matrix):
                D = 1 - self.distance_matrix
                np.fill_diagonal(D, 0)
                score = silhouette_score(D, labels, metric='precomputed')
                print(f"    silhouette score = {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_labels = labels
            else:
                print("    silhouette score = N/A (only one cluster or all noise)")

        print(f"\nBest eps selected: {best_eps} with silhouette = {round(best_score, 3)}\n")

        # Cluster with best eps
        dbscan = DBSCAN(eps=best_eps, min_samples=5, metric='precomputed')
        self.sample_labels = dbscan.fit_predict(self.distance_matrix)

        unique, counts = np.unique(self.sample_labels, return_counts=True)
        print("Cluster distribution in sample:")
        for u, c in zip(unique, counts):
            print(f"  Cluster {u}: {c} molecules")

        return self

    def map_to_full_dataset(self):
        # Map sampled clusters back to all molecules
        print("\nMapping all molecules to nearest sampled cluster...")
        n_total = len(self.fingerprints_morgan)
        self.all_labels = np.full(n_total, -1)

        for i in range(n_total):
            sims = BulkTanimotoSimilarity(self.fingerprints_morgan[i], self.sample_fps)
            if max(sims) > 0.3:
                nearest_sample = np.argmax(sims)
                self.all_labels[i] = self.sample_labels[nearest_sample]
            if i % 2000 == 0:
                print(f"Mapped {i}/{n_total}")

        np.save("cluster_labels_all.npy", self.all_labels)
        print("Saved all molecule cluster labels to cluster_labels_all.npy\n")
        return self

    def run(self):
        self.prepare_data()
        self.calculate_tanimoto_distance()
        self.optimize_dbscan()
        self.map_to_full_dataset()
        print("This method shows ~97% noise ratio due to sampling constraints.\n")
        return self


# ============================================================================
# 2: Molecular Descriptors + DBSCAN (Recommended)
# ============================================================================

class MolecularDescriptorClustering:

    # Clustering based on molecular descriptors using DBSCAN
    # Better performance compared to Morgan Fingerprint method

    def __init__(self, dataset):
        self.dataset = dataset
        self.X = None
        self.X_scaled = None
        self.X_pca = None
        self.n_components = None
        self.best_eps = None
        self.labels = None

    def prepare_features(self):
        # Extract and standardize molecular descriptors.
        print("=" * 70)
        print("METHOD 2: Molecular Descriptors + DBSCAN (Recommended)")
        print("=" * 70)

        # Extract molecular descriptors
        self.X = Molecular_Descriptors(self.dataset)

        # Select features
        X_features = self.X[['MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']]

        # Standardize
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X_features)

        print(f"Extracted {len(X_features.columns)} molecular descriptors")
        print(f"Total molecules: {len(self.X)}\n")
        return self

    def perform_pca(self, variance_threshold=0.9, show_plot=True):
        """Perform PCA for dimensionality reduction."""
        print("Performing PCA for dimensionality reduction...")

        # Full PCA to analyze variance
        pca_full = PCA().fit(self.X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)

        # Select number of components
        self.n_components = np.argmax(cumvar >= variance_threshold) + 1
        print(f"Selected {self.n_components} components (retains {variance_threshold * 100}% variance)\n")

        # Apply PCA
        pca = PCA(n_components=self.n_components, random_state=42)
        self.X_pca = pca.fit_transform(self.X_scaled)

        # Visualization
        if show_plot:
            plt.figure(figsize=(7, 5))
            plt.plot(cumvar, marker='o')
            plt.axhline(y=variance_threshold, color='r', linestyle='--',
                        label=f'{variance_threshold * 100}% threshold')
            plt.axvline(x=self.n_components - 1, color='g', linestyle='--',
                        label=f'Selected: {self.n_components}')
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title("PCA Explained Variance")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return self

    def optimize_dbscan(self, eps_values=[0.3, 0.4, 0.5, 0.6], show_results=True):
        """Find optimal eps parameter for DBSCAN."""
        print("Optimizing DBSCAN eps parameter:")

        sil_scores = []
        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=5)
            labels = db.fit_predict(self.X_pca)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            sil = silhouette_score(self.X_pca, labels) if n_clusters > 1 else -1
            sil_scores.append(sil)
            print(f"  eps={eps:.2f}, clusters={n_clusters}, silhouette={sil:.3f}")

        self.best_eps = eps_values[np.argmax(sil_scores)]
        print(f"\nOptimal eps selected: {self.best_eps:.2f}\n")
        return self

    def cluster(self):
        """Perform final clustering with optimal parameters."""
        print("Performing DBSCAN clustering...")
        dbscan = DBSCAN(eps=self.best_eps, min_samples=5)
        self.labels = dbscan.fit_predict(self.X_pca)
        self.X['Cluster'] = self.labels

        # Statistics
        unique, counts = np.unique(self.labels, return_counts=True)
        cluster_info = pd.DataFrame({'Cluster': unique, 'Count': counts})
        print("Cluster distribution:")
        print(cluster_info)
        print()

        return self

    def analyze_clusters(self):
        """Analyze cluster characteristics and activity."""
        print("Analyzing cluster characteristics...")

        # Average activity per cluster
        cluster_activity = self.X.groupby('Cluster')['HIV_active'].mean()
        print("\nAverage HIV activity per cluster:")
        print(cluster_activity)

        # Average descriptors per cluster
        desc_means = self.X.groupby('Cluster')[
            ['MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']
        ].mean()
        desc_means['HIV_active_ratio'] = cluster_activity
        print("\nAverage molecular descriptors per cluster:")
        print(desc_means)

        return cluster_activity, desc_means

    def visualize_clusters(self, cluster_activity, desc_means):
        # visualizations of clustering results.
        # 1. PCA scatter plot
        plt.figure(figsize=(8, 6))
        noise_mask = (self.labels == -1)
        plt.scatter(self.X_pca[noise_mask, 0], self.X_pca[noise_mask, 1],
                    c='red', s=10, label='Noise', alpha=0.5)
        plt.scatter(self.X_pca[~noise_mask, 0], self.X_pca[~noise_mask, 1],
                    c=self.labels[~noise_mask], cmap='viridis', s=15)
        plt.title(f"DBSCAN Clustering (eps={self.best_eps:.2f})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.colorbar(label='Cluster ID')
        plt.tight_layout()
        plt.show()

        # 2. Activity bar plot
        plt.figure(figsize=(8, 5))
        cluster_activity.plot(kind='bar', color='teal')
        plt.title("Average HIV Active Ratio per Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel("HIV Active Ratio")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # 3. Descriptor heatmap
        plt.figure(figsize=(10, 5))
        sns.heatmap(desc_means.T, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Value'})
        plt.title("Average Molecular Descriptors per Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel("Descriptor")
        plt.tight_layout()
        plt.show()

    def evaluate_as_classifier(self):
        """Evaluate clustering as activity predictor."""
        print("\n" + "=" * 70)
        print("Evaluating Clustering as Activity Predictor")
        print("=" * 70)

        # Define active clusters
        mean_activity = self.X['HIV_active'].mean()
        cluster_activity = self.X.groupby('Cluster')['HIV_active'].mean()
        active_clusters = cluster_activity[cluster_activity > mean_activity].index.tolist()

        print(f"Overall activity rate: {mean_activity:.3f}")
        print(f"Active clusters (activity > mean): {active_clusters}\n")

        # Create predictions
        self.X['Predicted_active'] = self.X['Cluster'].apply(
            lambda c: 1 if c in active_clusters else 0
        )

        # Confusion matrix
        cm = confusion_matrix(self.X['HIV_active'], self.X['Predicted_active'])
        print("Confusion Matrix:")
        print(cm)
        print()

        # Classification report
        print("Classification Report:")
        print(classification_report(self.X['HIV_active'], self.X['Predicted_active'], digits=3))

        # Visualize confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Inactive", "Active"])
        disp.plot(cmap='Blues')
        plt.title("Cluster-based HIV Activity Prediction")
        plt.tight_layout()
        plt.show()

        return self

    def run(self, show_plots=True):
        """Execute complete pipeline."""
        self.prepare_features()
        self.perform_pca(show_plot=show_plots)
        self.optimize_dbscan()
        self.cluster()
        cluster_activity, desc_means = self.analyze_clusters()

        if show_plots:
            self.visualize_clusters(cluster_activity, desc_means)

        self.evaluate_as_classifier()
        print("\n✅ Molecular descriptor clustering completed successfully!\n")
        return self




if __name__ == "__main__":
    # Uncomment to run Method 1 (Morgan Fingerprint - for reference)
    # print("\n" + "=" * 70)
    # print("Running Method 1: Morgan Fingerprint + DBSCAN")
    # print("=" * 70 + "\n")
    # method1 = MorganFingerprintClustering(dataset, sample_size=2000)
    # method1.run()

    # Run Method 2 (Molecular Descriptors - recommended)
    print("\n" + "=" * 70)
    print("Running Method 2: Molecular Descriptors + DBSCAN")
    print("=" * 70 + "\n")
    method2 = MolecularDescriptorClustering(dataset)
    method2.run(show_plots=True)

    # I also tried with Kmeans model, it can also divide the data into 4 clusters however it is not as detailed and accurate as DBSCAN
    # The images from this process are also attached in the path C:\python project\XiaolongYANG_ZigeHUANGCheminformatics-drug-design\Clustering\Figure\Kmeans
    # it serves as a reference as well