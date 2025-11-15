import pandas as pd

df = pd.read_csv("WineQT.csv")
print("Информация о датасете:")
print(df.info())

df = df.drop(columns=['Id'])

