# Telco Customer Churn Prediction System 

## Giới thiệu Dự án (Project Overview)
Dự án này ứng dụng Học máy (Machine Learning) để giải quyết bài toán cốt lõi trong ngành Viễn thông: **Dự đoán Khách hàng rời bỏ dịch vụ (Customer Churn Prediction)**. 

Thay vì sử dụng các chiến dịch Marketing giữ chân khách hàng một cách dàn trải và tốn kém, hệ thống này giúp doanh nghiệp nhận diện chính xác nhóm khách hàng có rủi ro rời đi cao nhất. Dự án bao gồm một Pipeline xử lý dữ liệu tự động, hệ thống tối ưu hóa mô hình, và một giao diện Web App tương tác (Streamlit) hỗ trợ nhân viên chăm sóc khách hàng ra quyết định theo thời gian thực.

## 🛠 Công nghệ sử dụng (Tech Stack)
* **Ngôn ngữ & Phân tích Dữ liệu:** Python 3.x, Pandas, NumPy
* **Trực quan hóa (EDA & Evaluation):** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Logistic Regression, Decision Tree, GridSearchCV)
* **Lưu trữ & Triển khai (Deployment):** Joblib, Streamlit

## Cấu trúc Thư mục (Project Structure)
```text
telco-customer-churn/
│
├── data/
│   ├── raw/               # Chứa file dữ liệu gốc (telco_customers_churn.csv)
│   └── processed/         # Chứa dữ liệu đã được làm sạch và chuẩn hóa (train.csv, test.csv)
│
├── core/                  # Mã nguồn lõi
│   ├── preprocessing.py   # Làm sạch, mã hóa (One-Hot), chuẩn hóa và chia tập (Train/Test)
│   ├── train.py           # Huấn luyện mô hình, tinh chỉnh tham số (Hyperparameter Tuning)
│   └── evaluate.py        # Xuất báo cáo hiệu năng, vẽ ROC, Confusion Matrix
│
├── models/                # Lưu trữ các đối tượng ML đã được huấn luyện (scaler.pkl, best_model.pkl)
├── images/                # Biểu đồ tự động xuất ra trong quá trình chạy Pipeline
│
├── app.py                 # Giao diện Web App tương tác dành cho người dùng cuối (Streamlit)
├── requirements.txt       # Danh sách thư viện môi trường
└── README.md              # Tài liệu dự án
```

## Hướng dẫn cài đặt và khởi chạy (Getting Started)

### Bước 1: Thiết lập môi trường ảo (Khuyến nghị)
Mở Terminal tại thư mục gốc của dự án và chạy các lệnh sau:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
venv\Scripts\activate
# Kích hoạt môi trường (Mac/Linux)
## source venv/bin/activate

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Chạy Luồng Dữ liệu Học máy
Thực thi tuần tự 3 tệp lệnh sau để trải nghiệm vòng đời huấn luyện mô hình:

```bash
# 1. Tiền xử lý dữ liệu và Trực quan hóa EDA
python core/preprocessing.py

# 2. Huấn luyện mô hình (Tìm kiếm tham số tối ưu với 5-Fold Cross Validation)
python core/train.py

# 3. Đánh giá mô hình trên tập Dữ liệu mới (Test Set)
python core/evaluate.py
```

### Bước 3: Khởi chạy Giao diện Web (Web App Deployment)
Chạy lệnh sau để mở giao diện Hệ thống Hỗ trợ Ra quyết định trên trình duyệt:

```bash
streamlit run app.py
```