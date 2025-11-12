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

df = pd.read_csv("processed_titanic1.csv")

df_k = pd.get_dummies(df, drop_first=True)
df_clean = df_k.dropna()

X = df_clean.drop(['Transported'], axis=1)
y = df_clean['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) #обучающая и тестовая

print(f"Обучающая выборка для Transported: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Тестовая выборка для Transported: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

#регрессия для age

X_a = df_clean.drop(['Age'], axis=1)
y_a = df_clean['Age']

X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size=0.4, random_state=42)

print(f"Обучающая выборка для Age: {X_a_train.shape[0]} samples ({X_a_train.shape[0]/len(X)*100:.1f}%)")
print(f"Тестовая выборка для Age: {X_a_test.shape[0]} samples ({X_a_test.shape[0]/len(X)*100:.1f}%)")

linear_model = LinearRegression()
linear_model.fit(X_a_train, y_a_train)
y_pred_test = linear_model.predict(X_a_test)

#оценка работы регрессионной модели

MSE = mean_squared_error(y_a_test, y_pred_test)
RMSE = root_mean_squared_error(y_a_test, y_pred_test)
MAE = mean_absolute_error(y_a_test, y_pred_test)

print(f"Среднеквадратичная ошибка: {MSE:.2f}")
print(f"Корень среднеквадратичной ошибки: {RMSE:.2f}")
print(f"Средняя абсолютная ошибка: {MAE:.2f}")

#задача классификации

logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
y_pred_test2 = logreg_model.predict(X_test)

#оценка работы классификационной модели

accuracy = accuracy_score(y_test, y_pred_test2)

print(f"Точность: {accuracy:.4f}")
print(f"Доля правильных ответов: {accuracy*100:.2f}%")

#матрица ошибок

cm = confusion_matrix(y_test, y_pred_test2)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#комплексный отчёт по классификационной модели

report = classification_report(y_test, y_pred_test2)
print(report)