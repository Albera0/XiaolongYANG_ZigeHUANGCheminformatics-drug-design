import pandas as pd
from rdkit import Chem


#Data loading function
def DataRead() :
    #Read the data from Lipophilicity.csv
    lipo_df = pd.read_csv('Dataset/Lipophilicity.csv')
    print("Lipophilicity Data: ", lipo_df.sample(3), "\n")

    #Check if there is missing data
    print(lipo_df.isna().sum())

    #Remove the missing data
    lipo_df = lipo_df[lipo_df.isna().sum(1)==0]
    print(lipo_df.isna().sum())

    #Get the smiles and octanol/water distribution coefficient
    smiles = lipo_df['smiles'].values
    y  = lipo_df['exp'].values
    
    #Check if the reading is finished
    print("smiles data: ", smiles[:3], "\n")
    print("octanol/water distribution coefficient data: ", y[:3], "\n")
    return lipo_df, smiles, y

lipo_df, smiles_list, y = DataRead()

#Canonicalization of the smiles
def CanonicalizeSmiles(smiles):
    #Take a non-canonical SMILES and returns the canonical version

    #create a mol object from input smiles
    mol = Chem.MolFromSmiles(smiles)

    #convert the previous mol object to SMILES using Chem.MolToSmiles()
    canonical_smiles = Chem.MolToSmiles(mol)

    return canonical_smiles

#create a new list  to list of non-canonical SMILES
canonical_smiles = [CanonicalizeSmiles(smiles) for smiles in smiles_list]

