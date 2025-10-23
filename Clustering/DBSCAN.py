import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity
from feature_representation import Morgan_fingerprints, dataset

fingerprints_morgan = Morgan_fingerprints(dataset.canonical_smiles)
def tanimoto_distance(fingerprints):
    n = len(fingerprints)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sims = BulkTanimotoSimilarity(fingerprints[i], fingerprints)
        dist_matrix[i, :] = [1 - s for s in sims]
        if i % 500 == 0:
            print(f"Processed {i}/{n} fingerprints")
    return dist_matrix

# calculate and save
print("Calculating Tanimoto distance matrix...")
Sim = tanimoto_distance(fingerprints_morgan)
np.save('tanimoto_distance_matrix.npy', Sim)
print("Saved distance matrix to tanimoto_distance_matrix.npy")

# load and verify
Sim_loaded = np.load('tanimoto_distance_matrix.npy')
print("Matrix loaded. Shape:", Sim_loaded.shape)
print("Symmetric check:", np.allclose(Sim_loaded, Sim_loaded.T, atol=1e-5))







# Clustering based on activity patterns