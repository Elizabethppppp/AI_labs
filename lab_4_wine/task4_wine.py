import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("WineQT.csv")
print("Информация о датасете:")
print(df.info())

df = df.drop(columns=['Id'])

df['label'] = (df['quality'] >= 6).astype(int)
X = df.drop(columns=['quality', 'label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)