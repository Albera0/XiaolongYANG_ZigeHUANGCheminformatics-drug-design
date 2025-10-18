# XiaolongYANG_ZigeHUANG_Cheminformatics-drug-design
This project was carried out as a course assignment in Chemoinformatics and Drug Design by YANG Xiaolong and HUANG Zige. By using datasets from the MoleculeNet benchmark suite, the project explores molecular property prediction. It consists of two main parts: regression prediction on a regression dataset from MoleculeNet, and classification prediction on a classification dataset.

## Regression analysis on the Lipophilicity dataset
### Project Structure
Project of Regression analysis mainly consists of the database, data reading, data preprocessing, prediction models, and result visualization. The models include a Random Forest model and a Message Passing Neural Network (MPNN).

#### Database
The database used is the Lipophilicity dataset, which includes molecular SMILES and experimental results of octanol/water distribution coefficients.

#### Data reading
The data loading module includes data reading function and canonicalization of the SMILES.

#### Data preprocessing
The data preprocessing module includes several commonly used molecular featurization strategies, such as Morgan fingerprints and molecular descriptors, as well as functions for removing zero-variance variables after analysis. It also contains a function for converting SMILES into graph data.

#### Prediction models


#### Result visualization