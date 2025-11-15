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

#(оценка работы регрессионной модели)
MSE = mean_squared_error(y_reg_test, y_reg_pred)
RMSE = root_mean_squared_error(y_reg_test, y_reg_pred)
MAE = mean_absolute_error(y_reg_test, y_reg_pred)

print(f"Среднеквадратичная ошибка: {MSE:.2f}")
print(f"Корень среднеквадратичной ошибки: {RMSE:.2f}")
print(f"Средняя абсолютная ошибка: {MAE:.2f}")

#(визуализация дерева)
plt.figure(figsize=(20, 10))
plot_tree(reg_tree, filled=True, feature_names=X_reg.columns, rounded=True)
plt.title("Дерево решений для регрессии alcohol")
plt.show()

#классификация для качества

df["good_quality"] = (df["quality"] >= 7).astype(int)

X_clf = df.drop(columns=["quality", "good_quality"])
y_clf = df["good_quality"]

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf)

#классифификация деревом
clf_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_tree.fit(X_clf_train, y_clf_train)
y_clf_proba = clf_tree.predict_proba(X_clf_test)[:, 1]

#рок кривая
fpr, tpr, _ = roc_curve(y_clf_test, y_clf_proba)
roc_auc = auc(fpr, tpr)
print(f"РОК кривая: {roc_auc:.4f}")

#визуализация
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC-кривая (good_quality)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

df.to_csv("wine3.csv", index=False)