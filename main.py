from DataRead_and_preprocessing import load_hiv_dataset
from feature_representation import Morgan_fingerprints


dataset = load_hiv_dataset('Dataset/HIV.csv')
fingerprints = Morgan_fingerprints(dataset.canonical_smiles)

print("Number of samples:", len(fingerprints))
print("Fingerprint length:", len(fingerprints[0]))
