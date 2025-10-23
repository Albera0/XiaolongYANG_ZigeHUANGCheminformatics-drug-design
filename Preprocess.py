import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from deepchem.feat import RDKitDescriptors
from sklearn.feature_selection import VarianceThreshold
import Read_Data

from tqdm import tqdm
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd


##Preprocess for the Random Forests
#Load the dataset
lipo_df, smiles_list, y = Read_Data.DataRead()

#Canonicalization of the smiles
canonical_smiles = [Read_Data.CanonicalizeSmiles(smiles) for smiles in smiles_list]

#Create the fingerprint
def Fingerprint(canonical_smiles):
    # Create a fingerprint generator and select the Morgan Fingerprint with
    # radius 2 and 2048 dimensions
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

    smiles_fp = []
    for smiles in canonical_smiles:
        #Change smiles into mol
        mol = Chem.MolFromSmiles(smiles)

        #Create the fingerprint
        smiles_fp.append(mfpgen.GetFingerprint(mol))

    #print vector length
    #This two part, ask GPT to get some adivce of reading the data
    print(smiles_fp[0].GetNumBits())

    #visualize vector as list
    print(smiles_fp[0].ToList())

    return smiles_fp

smiles_fp = Fingerprint(canonical_smiles)

#Create the molecular descriptors
def MolDescriptor(canonical_smiles):
    #Use molecular descriptors from RDKit
    featurizer = RDKitDescriptors()
    features = featurizer.featurize(canonical_smiles)
    print("Number of generated molecular descriptors:", features.shape[1])

    #Drop the features with invalid values
    features = features[:, ~np.isnan(features).any(axis=0)]
    print("Number of molecular descriptors without invalid values: ", features.shape[1])

    return features

features = MolDescriptor(canonical_smiles)

#Feature selection
def FeatSelection(features):
    #Removed all zero-variance features
    selector = VarianceThreshold(threshold=0.0)
    features = selector.fit_transform(features)
    print("Number of fratures after removing zero-variance features: ",
        features.shape[1])
    
    return features

sel_feature = FeatSelection(features)


##Preprocess for the Graph Convolutional Model
#Ask GPT how to pass my own dataset in the class
class GraphFeature(InMemoryDataset):

    def __init__(self, canonical_smiles, y_org, transform=None):
        self.canonical_smiles = canonical_smiles
        self.y_org = y_org
        super().__init__('.', transform)
        self.data, self.slices = self._process_in_memory()

    def _process_in_memory(self):
        #Builds Data directly in memory.
        data_list = []
        smiles = self.canonical_smiles
        target = self.y_org

        # Convert SMILES into graph data
        print('Converting SMILES strings into graphs...')
        data_list = []
        for i, smi in enumerate(tqdm(smiles)):

            # get graph data from SMILES
            graph = smiles2graph(smi)

            # convert to tensor and pyg data
            x = torch.tensor(graph['node_feat'], dtype=torch.long)
            edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
            y = torch.tensor([target[i]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        return self.collate(data_list)