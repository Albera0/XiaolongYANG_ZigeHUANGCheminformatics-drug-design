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




## Main Contributors
YANG Xiaolong and HUANG Zige are the main contributors to this project. Yang Xiaolong was mainly responsible for the regression analysis, while Huang Zige focused on the classification analysis. Data loading and preprocessing, as well as result visualization, were completed jointly by both.