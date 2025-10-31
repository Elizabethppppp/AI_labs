import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_csv("processed_titanic.csv")

df_k = pd.get_dummies(df, drop_first=True)
df_clean = df_k.dropna()

#регрессия для age

X_a = df_clean.drop(['Age'], axis=1)
y_a = df_clean['Age']

X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size=0.4, random_state=42)

print(f"Обучающая выборка для Age: {X_a_train.shape[0]} samples ({X_a_train.shape[0]/len(X_a)*100:.1f}%)")
print(f"Тестовая выборка для Age: {X_a_test.shape[0]} samples ({X_a_test.shape[0]/len(X_a)*100:.1f}%)")

#регрессия деревом
tree_regressor = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_regressor.fit(X_a_train, y_a_train)
y_pred_tree = tree_regressor.predict(X_a_test)

#оценка работы регрессионной модели

MSE = mean_squared_error(y_a_test, y_pred_tree)
RMSE = root_mean_squared_error(y_a_test, y_pred_tree)
MAE = mean_absolute_error(y_a_test, y_pred_tree)

print(f"Среднеквадратичная ошибка: {MSE:.2f}")
print(f"Корень среднеквадратичной ошибки: {RMSE:.2f}")
print(f"Средняя абсолютная ошибка: {MAE:.2f}")

#классификация для transported

X_t = df_clean.drop(['Transported'], axis=1)
y_t = df_clean['Transported']

X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(
    X_t, y_t, test_size=0.3, random_state=42, stratify=y_t
)

#классификация деревом

tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_t_train, y_t_train)
y_t_proba = tree_clf.predict_proba(X_t_test)

#рок кривая

fpr, tpr, _ = roc_curve(y_t_test, y_t_proba[:, 1])
roc_auc = auc(fpr, tpr)
print(f"ROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.show()


