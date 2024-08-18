import pandas as pd

df = pd.to_csv('./splits/dataset.csv')
df.drop(labels='Unnamed: 0', axis=1, inplace=True)
