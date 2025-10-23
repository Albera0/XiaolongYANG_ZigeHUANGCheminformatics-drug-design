import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem


class DatasetManager(Dataset):
    def __init__(self,csv_path,verbose=True):
        self.data = pd.read_csv(csv_path)

        # delete the missing data
        self.data = self.data.dropna()

        # locate SMILES
        self.smiles = self.data.iloc[:, 0].values

        # filter invalid SMILES
        self.verbose = verbose
        self.canonical_smiles = []
        self.valid_indices = []  #
        invalid_count = 0

        # canonicalize the SMILES
        for idx, s in enumerate(self.smiles):
            canonical = self.canonicalize_smiles(s)
            if canonical is not None:
                self.canonical_smiles.append(canonical)
                self.valid_indices.append(idx)
            else:
                invalid_count += 1

        if self.verbose and invalid_count > 0:
            print(f"   filter invalid SMILES: {invalid_count}  ({invalid_count / len(self.smiles) * 100:.2f}%)")

    # canonicalize the SMILES
    def canonicalize_smiles(self, smiles):
        # Canonicalize the SMILES string
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            return canonical_smiles
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            return None


    def __len__(self):
        return len(self.canonical_smiles)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]  # Original index
        smiles = self.canonical_smiles[idx]  # canonical SMILES
        label = self.data.iloc[real_idx]['HIV_active']
        return smiles, label


def load_hiv_dataset(csv_path='Dataset/HIV.csv'):
    return DatasetManager(csv_path)


# Debug the module on its own
if __name__ == "__main__":
    dataset = load_hiv_dataset()
    print(dataset[0])




