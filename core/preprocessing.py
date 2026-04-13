import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

## Tạo đường dẫn đến các thư mục
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

images_dir = os.path.join(project_root, 'images')
models_dir = os.path.join(project_root, 'models')
processed_dir = os.path.join(project_root, 'data', 'processed')
raw_data_path = os.path.join(project_root, 'data', 'raw', 'telco_customers_churn.csv')

## Tạo các thư mục images, models, data/processed nếu chưa có
os.makedirs(images_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)



sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.dpi"] = 120


#------------------------------------------------
### LOAD DATASET
#------------------------------------------------


print("--------------------------Bắt đầu quy trình tiền xử lý số liệu--------------------------")

## Mở file dataset nằm trong folder data/raw
try:
    df = pd.read_csv(raw_data_path)
    print("Mở file dữ liệu thành công !!")
except FileNotFoundError:
    print(f"Không tìm thấy file tại đường dẫn {raw_data_path}. Vui lòng kiểm tra lại !!")
    exit()


#------------------------------------------------
### CÁC BƯỚC LÀM SẠCH DỮ LIỆU
#------------------------------------------------


## Chuyển giá trị của đặc trưng TotalCharges thành dạng số (numeric), các giá trị không thể chuyển đổi gán giá trị NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

## Fill các giá trị NaN bằng giá trị 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.replace(["No internet service", "No phone service"], "No", inplace=True)

## Đưa giá trị cột Churn về tương ứng Yes -> 1 / No -> 0
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

## Xỏa bỏ cột customerID vì nó không mang lại information gain cho model
df = df.drop(columns=['customerID'])

## In ra tỷ lệ Churn so với No Churn trên toàn bộ tập dữ liệu (kiểm tra tính cân bằng của dữ liệu)
print(f"Tỷ lệ Churn: {df['Churn'].mean():.2%}")


#------------------------------------------------
### TRỰC QUAN HÓA DỮ LIỆU ĐỂ LỰA CHỌN CÁC ĐẶC TRƯNG PHÙ HỢP
#------------------------------------------------


## Vẽ biểu đồ tròn thể hiện tỷ lệ Churn sau đó lưu biểu đồ vào folder images
plt.figure(figsize=(6,6))
plt.pie(df['Churn'].value_counts(), labels=["No Churn","Churn"], autopct='%1.1f%%', startangle=90, shadow=True)
plt.title("Churn Distribution")
plt.savefig(os.path.join(images_dir, '01_churn_distribution.png'), bbox_inches='tight')
plt.close()

## Vẽ các biểu đồ cột của các đặc trưng không phải là numeric với tỷ lệ Churn cho từng giá trị của đặc trưng đó sau đó lưu biểu đồ vào folder images
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
plt.savefig(os.path.join(images_dir, '02_categorical_churn.png'), bbox_inches='tight')
plt.close()

## Vẽ biểu đồ phân phối của Churn với từng giá trị của các đặc trưng numeric sau đó lưu biểu đồ vào folder images
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
plt.savefig(os.path.join(images_dir, '03_numeric_distributions.png'), bbox_inches='tight')
plt.close()

## Vẽ biểu đồ tương quan giữa các đặc trưng với nhau sau đó lưu biểu đồ vào folder images
df_corr = df.copy()
for c in df_corr.select_dtypes(include=["object", "string"]).columns:
    df_corr[c] = LabelEncoder().fit_transform(df_corr[c])
corr = df_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14,10))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax, annot_kws={"size":7})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, '04_correlation_heatmap.png'), bbox_inches='tight')
plt.close()

print("Đã vẽ thành công và lưu toàn bộ các biểu đồ vào thư mục 'images/'")


#------------------------------------------------
### TRÍCH XUẤT VÀ MÃ HÓA CÁC ĐẶC TRƯNG (DỰA TRÊN CÁC BIỂU ĐỒ Ở TRÊN)
#------------------------------------------------


# Lọc bỏ các đặc trưng nhiễu (gender, MultipleLines và PhoneService) vì các đặc trưng này có hệ số tương quan rất thấp với label Churn nên không mang lại thông tin hữu ích (có thể gây nhiễu) 
# và đa cộng tuyến (TotalCharges và tenure có hệ số tương quan với nhau rất cao)
cols_to_drop = ['gender', 'TotalCharges', 'PhoneService', 'MultipleLines']
df_model = df.drop(columns=cols_to_drop).copy()

## Với các đặc trưng chứa giá trị nhị phân thì đưa về dạng 0 / 1 (như đã làm với Churn trước đó)
binary_cols = [c for c in df_model.select_dtypes(include=["object", "string"]).columns if set(df_model[c].unique()) <= {"Yes","No"}]
for c in binary_cols:
    df_model[c] = (df_model[c] == "Yes").astype(int)

## One-hot encoding với các đặc trưng phân loại thành các đặc trưng nhị phân
df_model = pd.get_dummies(df_model, drop_first=True)
print(f"Kích thước của bộ dữ liệu sau các bước tiền xử lý: {df_model.shape}")


#------------------------------------------------
### CHIA TẬP DỮ LIỆU VÀ CHUẨN HÓA DỮ LIỆU
#------------------------------------------------


## Tach cac feature thanh X va label thanh y
X = df_model.drop(columns=['Churn'])
y = df_model['Churn']

## Luu lai danh sach ten cac feature
feature_names = X.columns

## Chia thanh train va test set theo ti le 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

## Chuan hoa du lieu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Xuat ra file train va test
df_train_final = pd.DataFrame(X_train_scaled, columns=feature_names)
df_test_final = pd.DataFrame(X_test_scaled, columns=feature_names)
df_train_final['Churn'] = y_train.values
df_test_final['Churn'] = y_test.values
train_csv_path = os.path.join(processed_dir, 'train.csv')
test_csv_path = os.path.join(processed_dir, 'test.csv')
df_train_final.to_csv(train_csv_path, index=False)
df_test_final.to_csv(test_csv_path, index=False)

## Luu lai scaler
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

print("Hoàn tất quá trình tiền xử lý dữ liệu")
print(f"Đã lưu thành công các file:\n- {train_csv_path}\n- {test_csv_path}\n- {os.path.join(models_dir, 'scaler.pkl')}")