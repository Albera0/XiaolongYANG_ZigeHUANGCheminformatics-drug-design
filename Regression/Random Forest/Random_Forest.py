import sys 
sys.path.append("..") 
import Read_Data as rd
import Preprocess as prep

#Data Loading
lipo_df, smiles_list, y = rd.DataRead()

#create a new list  to list of non-canonical SMILES
canonical_smiles = [rd.CanonicalizeSmiles(smiles) for smiles in smiles_list]

#Create the fingerprint
smiles_fp = prep.Fingerprint(canonical_smiles)

#Create the molecular descriptors
features = prep.MolDescriptor(canonical_smiles)

#Feature selection
sel_fingerprint = prep.FeatSelection(smiles_fp)
sel_features = prep.FeatSelection(features)

#Random Forest Function, use function so can evaluate fueatures from
#fingerprint and molecular descriptors

#Dataset split


