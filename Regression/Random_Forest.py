import Read_Data as rd
import Preprocess as prep
from Draw_Figure import PredictedTrue
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

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

#Training model function
def TrainModel(model, X_train, y_train, X_test, y_test):
    # Train model
    model.fit(X_train, y_train)

    # Calculate RMSE
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    model_train_mse = mean_squared_error(y_train, y_pred_train)
    model_test_mse = mean_squared_error(y_test, y_pred_test)
    model_train_rmse = model_train_mse ** 0.5
    model_test_rmse = model_test_mse ** 0.5
    print(f"RMSE on train set: {model_train_rmse:.3f}, and test set: {model_test_rmse:.3f}.\n")
    

#Random Forest Function, use function so can evaluate fueatures from
#fingerprint/molecular descriptors
def RandomForest(features, featurization):
    #Dataset split
    X = features
    #Training data o.8 and test data size 0.2
    #Fixed seed using the random state
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0)

    #Data Standardization and Normalization
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    #Save origin data
    #Here ask GPT about how to save the origin data correctly
    X_train_ori = X_train.copy()
    X_test_ori = X_test.copy()

    #Transform data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    #Create the random forest regressor, using mean squared error (MSE) ascriterion
    rf_regress = RandomForestRegressor(n_estimators=10, random_state=0)

    # Train and test the random forest model
    print("Evaluating Random Forest Model.")
    TrainModel(rf_regress, X_train, y_train, X_test, y_test)

    #Cross-Validation and Hyperparameter Optimization
    #Hyperparameter
    param_grid = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [100, 150, 200, 250]
    }

    #Use 5-folds cross validation
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=0),
        param_grid,
        cv=5
    )
    grid_search.fit(X_train, y_train)

    #Re-train a model using best hyperparameters
    rf_gs = RandomForestRegressor(**grid_search.best_params_, random_state=0)

    print('Best paramters: ', grid_search.best_params_)
    print('Random forests performance after hyperparamter optimization:')
    TrainModel(rf_gs, X_train, y_train, X_test, y_test)

    #Visualization of the training results
    y_pred = rf_gs.predict(X_test)
    PredictedTrue(y_test, y_pred, featurization)
    
#Model training on different feaures
RandomForest(sel_fingerprint, "RF_Fingerprint") 
#RMSE on train set: 0.373, and test set: 0.850.
# Best paramters:  {'max_depth': 100, 'n_estimators': 250}
# Random forests performance after hyperparamter optimization:
# RMSE on train set: 0.310, and test set: 0.811.
print("Result of fingerprint above.")

RandomForest(sel_features, "RF_Molecular_Descriptors") 
#RMSE on train set: 0.320, and test set: 0.756.
# Best paramters:  {'max_depth': 100, 'n_estimators': 250}
# Random forests performance after hyperparamter optimization:
# RMSE on train set: 0.258, and test set: 0.714.
print("Result of molecular descriptors above.")


