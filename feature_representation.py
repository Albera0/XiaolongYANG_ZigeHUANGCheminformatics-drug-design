import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from DataRead_and_preprocessing import load_hiv_dataset
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import PandasTools
from rdkit.Chem.Descriptors import MolWt

dataset = load_hiv_dataset('C:\\Users\\Zoe\\Downloads\\HIV.csv')

'''
TEST
print("Total samples:", len(hiv_processed_data))
print("First 5 SMILES:", hiv_processed_data.canonical_smiles[:5])
print("Example data:", hiv_processed_data[0])

'''



# Create fingerprint generators(radius 2 and 2048 dimensions(bites)
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)



# Classical Fingerprints
# Feature Morgan
def Morgan_fingerprints(smiles_list,radius=2):

    fingerprints = []

    for i in smiles_list:
        mol = Chem.MolFromSmiles(i)
        if mol is not None:
            fp = mfpgen.GetFingerprint(mol)
            fingerprints.append(fp.ToList())
    return fingerprints

# RDkit
def RDkit_fingerprints(smiles_list):

    fingerprints = []

    for s in smiles_list:  # string
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = rdkgen.GetFingerprint(mol)
            fingerprints.append(fp.ToList())
    return fingerprints



#Molecular Desciptors
def Molecular_Descriptors(dataset):
    df = dataset.data.copy()
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles', molCol='Molecule')

    #Filter out invalid molecules (None)
    df = df[df['Molecule'].notnull()].copy()

    df['MW'] = df['Molecule'].apply(MolWt)
    calculator = MolecularDescriptorCalculator(['MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds'])
    properties = df['Molecule'].apply(calculator.CalcDescriptors)
    return pd.DataFrame(properties.tolist(), columns=['MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds'])



# Debug the module on its own
if __name__ == "__main__":

    # Compute Morgan fingerprints
    fingerprints_morgan = Morgan_fingerprints(dataset.canonical_smiles)
    print("Number of samples (Morgan):", len(fingerprints_morgan))
    print("Fingerprint length (Morgan):", len(fingerprints_morgan[0]))

    # Compute RDKit fingerprints
    fingerprints_rdkit = RDkit_fingerprints(dataset.canonical_smiles)
    print("Number of samples (RDKit):", len(fingerprints_rdkit))
    print("Fingerprint length (RDKit):", len(fingerprints_rdkit[0]))

    # Compute molecular descriptors
    desc = Molecular_Descriptors(dataset)
    print("Descriptors shape:", desc.shape)
    print(desc.head())




