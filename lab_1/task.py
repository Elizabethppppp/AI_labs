import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train.csv")

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

df['Cabin_Class'] = df['Cabin'].str[0]
df['Cabin_Class'] = df['Cabin_Class'].fillna('Unknown')
cabin_class_mapping = {
    'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'T': 0, 'Unknown': 0
}
df['Cabin_Class_Encoded'] = df['Cabin_Class'].map(cabin_class_mapping)

scaler = MinMaxScaler()
scaler.fit(df[['Cabin_Class_Encoded']])
df['Cabin_Class_Normalized'] = scaler.transform(df[['Cabin_Class_Encoded']].fillna(0))
print(df[['Cabin', 'Cabin_Class', 'Cabin_Class_Закодированный', 'Cabin_Class_Нормализованный']].head(10))