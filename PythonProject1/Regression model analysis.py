import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 讀取資料
df = pd.read_csv("data.csv")

# 特徵與目標欄位
features = ['water_inlet_temp', 'water_outlet_temp', 'wpump_power', 'oil_tank_temp']
target = 'outlet_temp'

X = df[features]
y = df[target]

# 資料標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 模型清單
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(kernel='rbf', C=10, epsilon=0.5)
}

results = []
all_predictions = {}

# 訓練與預測
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results.append({
        "Model": name,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape
    })

    all_predictions[name] = y_pred

# 轉換成 DataFrame
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print("模型評估結果：\n")
print(results_df)

# ----------- 🎨 可視化部分 -----------
# 真實值 vs 預測值散點圖
plt.figure(figsize=(14, 8))
for i, (name, preds) in enumerate(all_predictions.items()):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=y_test, y=preds, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("True Value")
    plt.ylabel("Predicted")
    plt.title(f"{name}")
plt.tight_layout()
plt.suptitle("True vs Predicted Values", fontsize=16, y=1.02)
plt.show()

# R² 比較長條圖
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="R2", y="Model", palette="coolwarm")
plt.title("R² Comparison")
plt.xlabel("R² Score")
plt.ylabel("Model")
plt.show()

# 模型誤差分布（Residual 分布）
plt.figure(figsize=(14, 8))
for i, (name, preds) in enumerate(all_predictions.items()):
    plt.subplot(2, 3, i+1)
    residuals = y_test - preds
    sns.histplot(residuals, kde=True, bins=25)
    plt.title(f"Residual Distribution: {name}")
    plt.xlabel("Error")
plt.tight_layout()
plt.show()
