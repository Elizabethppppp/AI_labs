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

df = pd.read_csv("WineQT.csv")

print("Первые 5 строк:")
print(df.head())
print("\nИнформация о датасете:")
print(df.info())

df['good_quality'] = (df['quality'] >= 7).astype(int)

df = df.drop(columns=['Id'])
print(df.info())

#для классификации
X = df.drop(columns=['quality','good_quality']) #то, на чём обучается модель
y = df['good_quality'] #то, что предсказываем

#для регрессии
X_reg = df.drop(columns=['alcohol'])
y_reg = df['alcohol']

#разделение на выборки для классификации
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

#разделение на выборки для регрессии
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.4, random_state=42)

print("Размер обучающей выборки:", X_train_r.shape)
print("Размер тестовой выборки:", X_test_r.shape)

#задача регрессии
regres_model = LinearRegression()
regres_model.fit(X_train_r, y_train_r)
y_pred_reg = regres_model.predict(X_test_r)

#оценка работы регрессионной модели

MSE = mean_squared_error(y_test_r, y_pred_reg)
RMSE = root_mean_squared_error(y_test_r, y_pred_reg)
MAE = mean_absolute_error(y_test_r, y_pred_reg)

print(f"Среднеквадратичная ошибка: {MSE:.2f}")
print(f"Корень среднеквадратичной ошибки: {RMSE:.2f}")
print(f"Средняя абсолютная ошибка: {MAE:.2f}")

#классификация
logreg_model = LogisticRegression(max_iter=1000, solver='liblinear')
logreg_model.fit(X_train, y_train)
y_pred_cl = logreg_model.predict(X_test)

#оценка работы классификационной модели

accuracy = accuracy_score(y_test, y_pred_cl)

print(f"Точность: {accuracy:.4f}")
print(f"Доля правильных ответов: {accuracy * 100:.2f}%")

#матрица ошибок
cm = confusion_matrix(y_test, y_pred_cl)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Истинные метки')
plt.xlabel('Предсказанные метки')
plt.show()

report = classification_report(y_test, y_pred_cl)
print("Отчёт по классификационной модели:")
print(report)