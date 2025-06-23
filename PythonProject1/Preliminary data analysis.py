import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1️⃣ 載入資料
file_path = "data.csv"
df = pd.read_csv(file_path)

# 2️⃣ 資料初探
print("📋 資料基本資訊")
print(df.info())
pd.set_option('display.max_columns', None)
print("\n📌 前幾筆資料")
print(df.head())
print("\n📊 敘述統計")
print(df.describe())

# 3️⃣ 檢查重複值與遺失值
print("\n🔍 重複值筆數：", df.duplicated().sum())
print("\n❗ 遺失值概況：")
print(df.isnull().sum())

# 4️⃣ 繪製直方圖與箱線圖（針對數值欄位）
sns.set_style('whitegrid')

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(12, 6 * len(numerical_columns)))  # 高度依欄位數自動調整

for i, col in enumerate(numerical_columns):
    # 直方圖
    plt.subplot(len(numerical_columns) * 2, 1, i * 2 + 1)
    sns.histplot(df[col], color='dodgerblue', bins=30, kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # 箱型圖
    plt.subplot(len(numerical_columns) * 2, 1, i * 2 + 2)
    sns.boxplot(x=df[col], color='salmon')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()

# 10️⃣ 繪製數值欄位的相關係數熱圖

# 👉 要排除的欄位（例如非數值或無意義特徵）
exclude_columns = ["id", "acmotor"] if "id" in df.columns and "acmotor" in df.columns else []

# 創建過濾後的 DataFrame，僅包含需要觀察的數值欄位
filtered_data = df.drop(columns=exclude_columns, errors='ignore')

# 僅選取數值型欄位計算相關係數
filtered_data = filtered_data.select_dtypes(include=["float64", "int64"])

# 計算相關矩陣
correlation_matrix = filtered_data[['rpm', 'gaccx', 'gaccy', 'gaccz', 'haccx', 'haccy', 'haccz']].corr()

# 🔥 繪製相關係數熱圖
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, square=True)
plt.title('Correlation Heatmap of Selected Numeric Variables')
plt.tight_layout()
plt.show()

#Filtered dataset excluding the specified columns
exclude_columns = ['id','acmotor','oilpump_power', 'wpump_outlet_press', 'gaccx', 'gaccy', 'gaccz', 'haccx', 'haccy', 'haccz', 'bearings', 'wpump', 'radiator','exvalve', 'acmotor','water_flow', 'rpm','air_flow', 'noise_db']

filtered_data = df.drop(columns=exclude_columns)

#Calculate the correlation matrix for the selected columns
correlation_matrix_selected = filtered_data.corr()

#Plotting the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, square=True)
plt.title('Correlation Map of Selected Variables')
plt.show()

# 5️⃣ 移除無意義欄位（視情況而定）
if "id" in df.columns:
    df = df.drop(columns=["id"])
if "acmotor" in df.columns:
    df = df.drop(columns=["acmotor"])

# 6️⃣ 特徵與標籤分離
label_columns = ["bearings", "wpump", "radiator", "exvalve"]
feature_columns = df.columns.difference(label_columns)
X = df[feature_columns]
y = df[label_columns]

# 7️⃣ 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8️⃣ 類別編碼
label_encoders = {}
y_encoded = pd.DataFrame()
for col in y.columns:
    le = LabelEncoder()
    y_encoded[col] = le.fit_transform(y[col])
    label_encoders[col] = le  # 儲存編碼器以備反解

# 9️⃣ 切分訓練集與測試集（80/20）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# 🔚 印出最終切分結果
print("\n✅ 資料切分完成")
print("訓練資料維度（X, y）:", X_train.shape, y_train.shape)
print("測試資料維度（X, y）:", X_test.shape, y_test.shape)
print("\n📌 標籤前五筆（已編碼）:")
print(y_encoded.head())
