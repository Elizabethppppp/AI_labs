import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
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

