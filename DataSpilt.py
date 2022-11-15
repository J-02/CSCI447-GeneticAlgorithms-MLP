
import pandas as pd

# getSamples
# --------------------------------------
# Splits dataset into 10 stratified samples
# input is a .data file in the /Data/ folder
# returns 10 stratified dataframes
def getSamples(dataset):
    df = pd.read_csv("Data/"+dataset,index_col=0, header=0)

    samples = []

# Creates samples for classification

    if 'class' in df.columns:

        # tuning sample completely randome
        # sample = df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=.1, replace=False))
        # df = df.drop(sample.index)
        # tune = sample

        # gets 10 samples of stratified data
        for i in range(10):

            n = 1/(10-i)
            sample = df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=n, replace=False))
            df = df.drop(sample.index)
            samples.append(sample)


# Creates samples for regression

    else:
        df = df.sort_values(by=df.columns[-1])
        # sample = df.iloc[0::10, :]
        # df2 = df.drop(sample.index)
        # tune = sample

        # gets 10 samples of stratified data
        for i in range(10):
            sample = df.iloc[i::10, :]
            samples.append(sample)


    return samples



