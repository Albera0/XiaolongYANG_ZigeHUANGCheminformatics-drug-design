import pandas as pd

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
    return smiles, y

simles, y = DataRead()
