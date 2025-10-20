import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train.csv")

print(df.head())

df.info()

colls=df.columns
print(colls)

