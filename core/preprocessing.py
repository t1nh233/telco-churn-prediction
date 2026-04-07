import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------------------------------------
# 0. THIẾT LẬP MÔI TRƯỜNG & THƯ MỤC
# ---------------------------------------------------------
# Xác định thư mục chứa file script này (thư mục 'core')
current_dir = os.path.dirname(os.path.abspath(__file__))

# Lùi lại 1 cấp để ra thư mục gốc của toàn bộ dự án
project_root = os.path.dirname(current_dir)

# Tạo các đường dẫn tuyệt đối chuẩn xác
image_dir = os.path.join(project_root, 'images')
models_dir = os.path.join(project_root, 'models')
processed_dir = os.path.join(project_root, 'data', 'processed')
raw_data_path = os.path.join(project_root, 'data', 'raw', 'telco_customers_churn.csv')

# Tạo thư mục (nằm ở cấp ngoài cùng của dự án)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.dpi"] = 120

print("Bắt đầu quy trình tiền xử lý dữ liệu...")

# ---------------------------------------------------------
# 1. LOAD DỮ LIỆU
# ---------------------------------------------------------
# Thay đổi đường dẫn này trỏ tới file csv trên máy tính của em
try:
    df = pd.read_csv(raw_data_path)
    print("✅ Load thành công dữ liệu!")
except FileNotFoundError:
    print(f"❌ Không tìm thấy file dữ liệu tại {raw_data_path}. Vui lòng kiểm tra lại!")
    exit()

# ---------------------------------------------------------
# 2. LÀM SẠCH DỮ LIỆU CƠ BẢN (DATA CLEANSING)
# ---------------------------------------------------------
# Xử lý TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Đưa Churn về 0/1
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Xóa customerID
df = df.drop(columns=['customerID'])

print(f"Tỷ lệ Churn chung: {df['Churn'].mean():.2%}")

# ---------------------------------------------------------
# 3. TRỰC QUAN HÓA DỮ LIỆU (EDA) & LƯU ẢNH
# ---------------------------------------------------------
print("Đang vẽ và lưu các biểu đồ EDA...")

# 3.1. Churn Distribution (Pie Chart)
plt.figure(figsize=(6,6))
plt.pie(df['Churn'].value_counts(), labels=["No Churn","Churn"], autopct='%1.1f%%', startangle=90, shadow=True)
plt.title("Churn Distribution")
plt.savefig(os.path.join(image_dir, '01_churn_distribution.png'), bbox_inches='tight')
plt.close() # Đóng plot để giải phóng RAM

# 3.2. Categorical Features Bar Charts
cat_cols = df.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges', 'tenure']).columns
n_cols = len(cat_cols)
n_rows = (n_cols + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ct_df = df.groupby(col)["Churn"].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
    ct_df['mean'].plot(kind="bar", ax=axes[i], color=sns.color_palette("husl", len(ct_df)), edgecolor="black", width=0.6)
    axes[i].set_title(f"Churn Rate by {col}", fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Churn Rate")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=25)
    axes[i].set_ylim(0, ct_df['mean'].max() + 0.1)
    
    for k, bar in enumerate(axes[i].patches):
        rate = bar.get_height()
        count = int(ct_df['count'].iloc[k])
        axes[i].text(bar.get_x() + bar.get_width()/2, rate + 0.01, f"{rate:.1%}\n({count})", ha="center", va="bottom", fontsize=9)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Churn Rate & Sample Size Across Categorical Features", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(image_dir, '02_categorical_churn.png'), bbox_inches='tight')
plt.close()

# 3.3. Numeric Feature Distributions by Churn
num_cols = ["tenure","MonthlyCharges","TotalCharges"]
fig, axes = plt.subplots(1, 3, figsize=(15,4))
for ax, col in zip(axes, num_cols):
    for label, grp in df.groupby("Churn"):
        grp[col].plot.kde(ax=ax, label=["No Churn","Churn"][label], linewidth=2)
    ax.set_title(f"Distribution of {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.legend()
plt.suptitle("Numeric Feature Distributions by Churn", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, '03_numeric_distributions.png'), bbox_inches='tight')
plt.close()

# 3.4. Correlation Heatmap
df_corr = df.copy()
for c in df_corr.select_dtypes(include=["object", "string"]).columns:
    df_corr[c] = LabelEncoder().fit_transform(df_corr[c])
corr = df_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14,10))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax, annot_kws={"size":7})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, '04_correlation_heatmap.png'), bbox_inches='tight')
plt.close()

print("✅ Đã lưu toàn bộ hình ảnh vào thư mục 'images/'")

# ---------------------------------------------------------
# 4. FEATURE ENGINEERING & ENCODING
# ---------------------------------------------------------
# Lọc bỏ các đặc trưng nhiễu và đa cộng tuyến
cols_to_drop = ['gender', 'TotalCharges', 'PhoneService', 'MultipleLines']
df_model = df.drop(columns=cols_to_drop).copy()

# Mã hóa nhị phân
binary_cols = [c for c in df_model.select_dtypes(include=["object", "string"]).columns if set(df_model[c].unique()) <= {"Yes","No"}]
for c in binary_cols:
    df_model[c] = (df_model[c] == "Yes").astype(int)

# One-Hot Encoding
df_model = pd.get_dummies(df_model, drop_first=True)
print(f"Kích thước ma trận X sau khi tối ưu: {df_model.shape}")

# ---------------------------------------------------------
# 5. CHIA TẬP, CHUẨN HÓA VÀ LƯU TRỮ DƯỚI DẠNG CSV
# ---------------------------------------------------------
# Tách Features (X) và Target (y)
X = df_model.drop(columns=['Churn'])
y = df_model['Churn']

# Lưu lại danh sách tên cột để dùng sau khi chuẩn hóa
feature_names = X.columns

# Chia Train/Test (Chia xong chúng vẫn khớp nhau 100% theo index)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Chuẩn hóa dữ liệu (Lưu ý: Đầu ra của scaler là Numpy Array)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- BƯỚC LẮP RÁP LẠI THÀNH DATAFRAME ĐỂ XUẤT CSV ---
# 1. Biến mảng Numpy thành DataFrame, gắn lại tên cột
df_train_final = pd.DataFrame(X_train_scaled, columns=feature_names)
df_test_final = pd.DataFrame(X_test_scaled, columns=feature_names)

# 2. Dán cột nhãn (y) vào lại (Phải dùng .values để bỏ index cũ, tránh lệch dòng)
df_train_final['Churn'] = y_train.values
df_test_final['Churn'] = y_test.values

# 3. Xuất ra file CSV
train_csv_path = os.path.join(processed_dir, 'train.csv')
test_csv_path = os.path.join(processed_dir, 'test.csv')

df_train_final.to_csv(train_csv_path, index=False)
df_test_final.to_csv(test_csv_path, index=False)

# Vẫn phải lưu scaler.pkl để ngày mai dự đoán khách hàng mới
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

print("✅ Hoàn tất toàn bộ Data Pipeline!")
print(f"Đã xuất file:\n- {train_csv_path}\n- {test_csv_path}\n- {os.path.join(models_dir, 'scaler.pkl')}")