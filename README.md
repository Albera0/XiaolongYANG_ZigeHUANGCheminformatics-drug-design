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
The modeling module consists of a Random Forest and a Message Passing Neural Network (MPNN). The Random Forest model is optimized through the cross-validation and hyperparameter optimization, whereas the MPNN is optimized through the use of skip connections and dropout techniques.
#### Result visualization
The result visualization module includes commonly used plots comparing predicted&true values, as well as training&validation loss plots to assess whether the network has converged.

### Instructions for Use
1.Place your database in the **Dataset** folder.

2.Modify the data reading path for **lipo_df** in the **DataRead** function of the **Read_Data** file, and update **smiles** and **y** to correspond to the column names for molecular SMILES and training data in your database.

3.To perform Random Forest prediction, run the **Random_Forests** file in the **Regression** folder. The visualization results will be saved in the **Figure** folder.

4.To perform Message Passing Neural Network prediction, run the **Graph_Convolutional_Model** file in the **Regression** folder. The visualization results will be saved in the **Figure** folder.

## Clustering analysis on the HIV Dataset
### Project Structure
Project of clustering analysis mainly consists of the database, data reading, data preprocessing, prediction models, and result visualization. The models include PCA+DBSCAN and a Autoencoder+HDBSCAN.
#### Database
The database used is the HIV dataset, which includes molecular SMILES , one-hot labels and activity.
#### Data reading
The data loading module includes data reading function and canonicalization of the SMILES.
#### Data preprocessing
The data preprocessing module includes several commonly used molecular featurization strategies, such as Morgan fingerprints and molecular descriptors, as well as Standardization and scaling of numerical features for neural network training.
#### Prediction models
The modeling module consists of a PCA+DBSCAN and a Autoencoder+HDBSCAN. The modeling module consists of a PCA + DBSCAN and an Autoencoder + HDBSCAN approach. The first one, , uses principal component analysis to reduce the feature dimensions of molecules, and then applies DBSCAN to group them based on their similarity. It gives a simple and clear way to observe molecular clusters. The second one, Autoencoder + HDBSCAN, replaces PCA with an autoencoder network that learns the most important features automatically. After compression, HDBSCAN is used to find clusters with more flexible shapes and clearer boundaries.
#### Result evaluation
The clusters are then put into more detalied analysis including Murcko scaffold, descriptors extraction etc, which provides the links between different functional groups and HIV_active, can be used as hint for detailed HIV detection.

### Instructions for Use
1.Place your database in the **Dataset** folder.

2.Modify the data reading path for **HIV** in the **DataManager** class of the **Read_Data** file.

3.Run "preprocess" file to get Molecular Descriptors and MorganFingerprint

4.To perform PCA+DBSCAN, run the **DBSCAN** file in the **Clustering** folder. The visualization results will displayed instantly.

5.To perform Autoencoder + HDBSCAN, run the **Autoencoder** file in the **Clustering** folder. The visualization results will be saved in the **Figure** folder.

6.Training logs are saved in the "logs\Autoencoder_experiment" folder. Checkpoints are saved in "Models".

## Main Contributors
YANG Xiaolong and HUANG Zige are the main contributors to this project. Yang Xiaolong was mainly responsible for the regression analysis, while Huang Zige focused on the classification analysis. Data loading and preprocessing, as well as result visualization, were completed jointly by both.
