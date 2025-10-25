import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset


# Data loading function
def DataRead():
    # Read the data from Lipophilicity.csv
    lipo_df = pd.read_csv('Dataset/Lipophilicity.csv')
    print("Lipophilicity Data: ", lipo_df.sample(3), "\n")

    # Check if there is missing data
    print(lipo_df.isna().sum())

    # Remove the missing data
    lipo_df = lipo_df[lipo_df.isna().sum(1) == 0]
    print(lipo_df.isna().sum())

    # Get the smiles and octanol/water distribution coefficient
    smiles = lipo_df['smiles'].values
    y = lipo_df['exp'].values

    # Check if the reading is finished
    print("smiles data: ", smiles[:3], "\n")
    print("octanol/water distribution coefficient data: ", y[:3], "\n")
    return lipo_df, smiles, y


lipo_df, smiles_list, y = DataRead()


# Canonicalization of the smiles
def CanonicalizeSmiles(smiles):
    # Take a non-canonical SMILES and returns the canonical version

    # create a mol object from input smiles
    mol = Chem.MolFromSmiles(smiles)

    # convert the previous mol object to SMILES using Chem.MolToSmiles()
    canonical_smiles = Chem.MolToSmiles(mol)

    return canonical_smiles


# create a new list  to list of non-canonical SMILES
canonical_smiles = [CanonicalizeSmiles(smiles) for smiles in smiles_list]


class DatasetManager(Dataset):
    def __init__(self, csv_path, verbose=True):
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


class DatasetFilter:
    # new added for filtering and balance the data in case of RAM runout

    def __init__(self, dataset):
        self.dataset = dataset
        self.df = dataset.data.copy()

    def get_positive_samples(self):
        # extract positive sample
        positive_df = self.df[self.df["HIV_active"] == 1].reset_index(drop=True)
        print(f"[INFO] Extracted {len(positive_df)} positive samples.")
        return positive_df

    def get_negative_samples(self):
        # negative sample
        negative_df = self.df[self.df["HIV_active"] == 0].reset_index(drop=True)
        print(f"[INFO] Extracted {len(negative_df)} negative samples.")
        return negative_df

    def get_balanced_dataset(self, ratio=1, random_state=42):
        # AI generated balanced dataset
        pos_df = self.get_positive_samples()
        neg_df = self.get_negative_samples()

        n_pos = len(pos_df)
        n_neg = int(ratio * n_pos)

        neg_sampled = neg_df.sample(n=min(n_neg, len(neg_df)), random_state=random_state)
        balanced_df = pd.concat([pos_df, neg_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        print(f"[INFO] Created balanced dataset: {len(balanced_df)} samples "
              f"(positive={len(pos_df)}, negative={len(neg_sampled)})")
        return balanced_df

    def summary(self):
        # print label
        counts = self.df["HIV_active"].value_counts().to_dict()
        total = len(self.df)
        print(f"[INFO] Dataset summary: {total} samples")
        for label, count in counts.items():
            print(f"    Label {label}: {count} ({count / total * 100:.2f}%)")


def load_hiv_dataset(csv_path='Dataset/HIV.csv'):
    return DatasetManager(csv_path)


# Debug the module on its own
if __name__ == "__main__":
    dataset = load_hiv_dataset()
    print(dataset[0])




