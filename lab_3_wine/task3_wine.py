import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

df = pd.read_csv("WineQT.csv")

df = df.drop("Id", axis=1)

#регрессия для алкоголя

target_reg = "alcohol"
X_reg = df.drop(columns=[target_reg])
y_reg = df[target_reg]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

#(регрессия деревом)
reg_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
reg_tree.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_tree.predict(X_reg_test)

#оценка работы регрессионной модели
MSE = mean_squared_error(y_reg_test, y_reg_pred)
RMSE = root_mean_squared_error(y_reg_test, y_reg_pred)
MAE = mean_absolute_error(y_reg_test, y_reg_pred)

print(f"Среднеквадратичная ошибка: {MSE:.2f}")
print(f"Корень среднеквадратичной ошибки: {RMSE:.2f}")
print(f"Средняя абсолютная ошибка: {MAE:.2f}")