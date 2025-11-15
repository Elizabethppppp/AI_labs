import pandas as pd

df = pd.read_csv("WineQT.csv")
print("Информация о датасете:")
print(df.info())

df = df.drop(columns=['Id'])

df['label'] = (df['quality'] >= 6).astype(int)
X = df.drop(columns=['quality', 'label'])
y = df['label']