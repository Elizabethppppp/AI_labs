import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv("processed_titanic.csv")

X = df.drop(['Transported'], axis=1)
y = df['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) #обучающая и тестовая

print(f"Обучающая выборка для Transported: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Тестовая выборка для Transported: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

#регрессия для age

X_a = df.drop(['Age'], axis=1)
y_a = df['Age']

X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size=0.4, random_state=42)

print(f"Обучающая выборка для Age: {X_a_train.shape[0]} samples ({X_a_train.shape[0]/len(X)*100:.1f}%)")
print(f"Тестовая выборка для Age: {X_a_test.shape[0]} samples ({X_a_test.shape[0]/len(X)*100:.1f}%)")