import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train.csv").head(500)

print(df.head())

df.info()

colls=df.columns
print(colls)

nul_matrix = df.isnull()
print(nul_matrix.sum())

cabin_mode = df['Cabin'].mode()[0]
home_mode = df['HomePlanet'].mode()[0]
sleep_mode = df['CryoSleep'].mode()[0]
dest_mode = df['Destination'].mode()[0]
vip_mode = df['VIP'].mode()[0]

room_median = df['RoomService'].median()

age_mean = df['Age'].mean()

df['Cabin'].fillna(cabin_mode, inplace=True)
df['HomePlanet'].fillna(home_mode, inplace=True)
df['CryoSleep'].fillna(sleep_mode, inplace=True)
df['Destination'].fillna(dest_mode, inplace=True)
df['VIP'].fillna(vip_mode, inplace=True)

df['RoomService'].fillna(room_median, inplace=True)

df['Age'].fillna(age_mean, inplace=True)

nul_matrix = df.isnull()
print(nul_matrix.sum())

scaler = MinMaxScaler()
df['Age_Normalized'] = scaler.fit_transform(df[['Age']])
print(f"\nНормализованный возраст:")
print(df[['Age', 'Age_Normalized']].head(10))

df = pd.get_dummies(df, columns=['Transported'], drop_first=True)
print("\nДанные после преобразования Transported:")
print(df[['Transported_True']].head(10))

columns_to_drop = ['Cabin', 'Destination', 'VIP', 'RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'CryoSleep', 'HomePlanet']
df = df.drop(columns=columns_to_drop)
print(f"Колонки после удаления:")
print(df.columns.tolist())

print(df.isnull().sum())

df.to_csv("processed_titanic.csv", index=False)
df.to_csv("processed_titanic1.csv", index=False)