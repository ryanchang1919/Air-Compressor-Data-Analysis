# 匯入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# 讀取資料
df = pd.read_csv('data.csv')  # ← 請確認檔案與此程式碼位於同一資料夾

# 建立分類目標欄位（將 oil_tank_temp 分為三類）
quantiles = df['oil_tank_temp'].quantile([0.33, 0.66]).values
df['oil_tank_temp_class'] = pd.cut(df['oil_tank_temp'],
                                   bins=[-np.inf, quantiles[0], quantiles[1], np.inf],
                                   labels=[0, 1, 2]).astype(int)

# 選取特徵與目標
features = ['outlet_temp', 'water_inlet_temp', 'water_outlet_temp', 'wpump_power']
X = df[features]
y = df['oil_tank_temp_class']

# 資料標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 模型設定
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 預備結果儲存區
results = {}
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# 模型訓練與預測
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro'),
        "Recall": recall_score(y_test, y_pred, average='macro'),
        "F1-score": f1_score(y_test, y_pred, average='macro'),
        "AUC-ROC": roc_auc_score(y_test_bin, y_proba, multi_class='ovr'),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True),
        "Probabilities": y_proba,
        "Predictions": y_pred
    }

# 顯示指標
for name, metrics in results.items():
    print(f"\n==== {name} ====")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-score: {metrics['F1-score']:.4f}")
    print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")
    print("Confusion Matrix:")
    print(metrics['Confusion Matrix'])

# 📊 混淆矩陣視覺化
plt.figure(figsize=(15, 4))
for idx, (name, metrics) in enumerate(results.items()):
    plt.subplot(1, 3, idx + 1)
    sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
plt.tight_layout()
plt.show()

# 📈 多類別 ROC 曲線（One-vs-Rest）
plt.figure(figsize=(8, 6))

for name, metrics in results.items():
    y_score = metrics["Probabilities"]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):  # 3 個類別
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(3):
        plt.plot(fpr[i], tpr[i],
                 label=f'{name} - Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves (OvR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 特徵重要性分析（使用 Random Forest）
importances = models['Random Forest'].feature_importances_
feature_names = features
indices = np.argsort(importances)[::-1]  # 特徵按重要性排序

plt.figure(figsize=(8, 6))
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()