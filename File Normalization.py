import pandas as pd
import os

for file in os.listdir("Data/"):
    if file.endswith(".data"):
        df = pd.read_csv("Data/"+file, index_col=0, header=0)
        output = df.iloc[:,-1]
        df = df.iloc[:,:-1]
        #print(df)
        normalized_df = (df - df.min()) / (df.max() - df.min())
        df = normalized_df.join(output)
        df.to_csv("Data/"+file)
