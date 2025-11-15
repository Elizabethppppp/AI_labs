import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("WineQT.csv")
print("Информация о датасете:")
print(df.info())

df = df.drop(columns=['Id'])

df['label'] = (df['quality'] >= 6).astype(int)
X = df.drop(columns=['quality', 'label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#задачa классификации методом случайного леса
rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=42, n_jobs=-1)

rf.fit(X_train, y_train)

print("OOB оценка:", rf.oob_score_)
print("OOB ошибка:", 1 - rf.oob_score_)

#(точность)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Точность случайного леса:", rf_acc)

rf_proba = rf.predict_proba(X_test)[:, 1]

#задачa классификации методом AdaBoos
adb = AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=42)

adb.fit(X_train_scaled, y_train)

#(точность)
adb_pred = adb.predict(X_test_scaled)
adb_acc = accuracy_score(y_test, adb_pred)
print("Точность AdaBoost:", adb_acc)

adb_proba = adb.predict_proba(X_test_scaled)[:, 1]

#задачa классификации методом методом градиентного бустинга
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)

gb.fit(X_train, y_train)

#(точность)
gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print("Точность градиентного бустинга:", gb_acc)

gb_proba = gb.predict_proba(X_test)[:, 1]

#ROC кривая
plt.figure(figsize=(10, 7))

#для случайного леса
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)

#для AdaBoost
fpr_adb, tpr_adb, _ = roc_curve(y_test, adb_proba)
roc_auc_adb = auc(fpr_adb, tpr_adb)

#для градиентного бустинга
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_proba)
roc_auc_gb = auc(fpr_gb, tpr_gb)

plt.plot(fpr_rf, tpr_rf, label=f"Случайный лес (точность = {roc_auc_rf:.3f})", linewidth=2)
plt.plot(fpr_adb, tpr_adb, label=f"AdaBoost (точность = {roc_auc_adb:.3f})", linewidth=2)
plt.plot(fpr_gb, tpr_gb, label=f"Градиентный бустинг (точность = {roc_auc_gb:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.title("ROC кривые", fontsize=14)
plt.xlabel("Доля ложных положительных результатов")
plt.ylabel("Доля истинно положительных результатов")
plt.legend()
plt.grid()
plt.show()

df.to_csv("wine4.csv", index=False)