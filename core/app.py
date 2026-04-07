import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# 1. CẤU HÌNH GIAO DIỆN
# ---------------------------------------------------------
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📡 Hệ thống Dự đoán Khách hàng Rời bỏ")
st.markdown("Nhân viên CSKH vui lòng nhập thông tin của khách hàng mới vào biểu mẫu dưới đây để hệ thống AI đánh giá rủi ro.")

# ---------------------------------------------------------
# 2. LOAD MÔ HÌNH & SCALER
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model = joblib.load(os.path.join(project_root, 'models', 'best_logistic_model.pkl'))
    scaler = joblib.load(os.path.join(project_root, 'models', 'scaler.pkl'))
    return model, scaler

try:
    model, scaler = load_models()
    # Lấy danh sách 26 cột chuẩn mà mô hình đòi hỏi
    expected_columns = scaler.feature_names_in_
except Exception as e:
    st.error("❌ Không tìm thấy mô hình. Hãy chắc chắn em đã chạy file train.py!")
    st.stop()

# ---------------------------------------------------------
# 3. GIAO DIỆN NHẬP LIỆU (FORM)
# ---------------------------------------------------------
with st.form("customer_input_form"):
    st.subheader("Thông tin Khách hàng")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Thông tin Dịch vụ Chính**")
        tenure = st.number_input("Thời gian sử dụng (Tháng)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Cước phí hàng tháng ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.5)
        contract = st.selectbox("Loại Hợp đồng", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Dịch vụ Internet", ["No", "DSL", "Fiber optic"]) # Chuyển No lên đầu
        payment = st.selectbox("Phương thức thanh toán", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    with col2:
        st.markdown("**Dịch vụ Cộng thêm**")
        # Thêm index=1 để Yes đứng trước nhưng mặc định tick vào No
        online_security = st.radio("Bảo mật Online (Online Security)", ["Yes", "No"], index=1)
        tech_support = st.radio("Hỗ trợ Kỹ thuật (Tech Support)", ["Yes", "No"], index=1)
        online_backup = st.radio("Sao lưu Online (Online Backup)", ["Yes", "No"], index=1)
        device_protection = st.radio("Bảo vệ Thiết bị (Device Protection)", ["Yes", "No"], index=1)

    with col3:
        st.markdown("**Thông tin Cá nhân & Giải trí**")
        senior_citizen = st.radio("Là người cao tuổi? (Senior Citizen)", ["Yes", "No"], index=1)
        partner = st.radio("Có đối tác/vợ chồng? (Partner)", ["Yes", "No"], index=1)
        dependents = st.radio("Có người phụ thuộc? (Dependents)", ["Yes", "No"], index=1)
        
        # Riêng Hóa đơn điện tử thường đa số khách hàng dùng, em có thể để index=0 (Tick Yes)
        paperless = st.radio("Hóa đơn điện tử? (Paperless Billing)", ["Yes", "No"], index=1) 
        
        streaming_tv = st.radio("Xem TV Trực tuyến? (Streaming TV)", ["Yes", "No"], index=1)

    # Nút bấm Gửi dữ liệu
    submitted = st.form_submit_button("Phân tích Rủi ro bằng AI", use_container_width=True)

# ---------------------------------------------------------
# 4. XỬ LÝ DỮ LIỆU (PREPROCESSING MỘT CÁCH THÔNG MINH)
# ---------------------------------------------------------
if submitted:
    # Bước 4.1: Tạo một DataFrame 1 dòng, chứa toàn số 0 với đúng 26 cột chuẩn
    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    # Bước 4.2: Gán các giá trị Số (Numeric)
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    
    # Bước 4.3: Gán các biến Nhị phân (Yes/No)
    # Hàm con để chuyển Yes/No thành 1/0
    def yes_no_to_int(val):
        return 1 if val == "Yes" else 0
        
    binary_mapping = {
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PaperlessBilling': paperless,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'StreamingTV': streaming_tv
    }
    
    for col_name, val in binary_mapping.items():
        if col_name in input_df.columns:
            input_df[col_name] = yes_no_to_int(val)
            
    # Bước 4.4: Bật các biến One-Hot Encoding
    # Nếu người dùng chọn Contract = "One year", ta tìm xem có cột "Contract_One year" không và gán nó = 1
    if f'Contract_{contract}' in input_df.columns:
        input_df[f'Contract_{contract}'] = 1
        
    if f'InternetService_{internet}' in input_df.columns:
        input_df[f'InternetService_{internet}'] = 1
        
    if f'PaymentMethod_{payment}' in input_df.columns:
        input_df[f'PaymentMethod_{payment}'] = 1

    # ---------------------------------------------------------
    # 5. CHUẨN HÓA VÀ DỰ ĐOÁN
    # ---------------------------------------------------------
    # Scale dữ liệu (Đầu ra bị mất tên cột, biến thành Numpy Array)
    scaled_array = scaler.transform(input_df)
    
    # Kỹ thuật FIX CẢNH BÁO: Bọc nó lại thành DataFrame và gắn lại tên cột
    scaled_df = pd.DataFrame(scaled_array, columns=expected_columns)
    
    # Lấy xác suất Churn (Đưa DataFrame vào để mô hình không kêu ca)
    churn_prob = model.predict_proba(scaled_df)[0][1]
    
    st.markdown("---")
    st.header("Kết quả Dự đoán")
    
    # Thanh hiển thị (Progress bar) cho sinh động
    st.progress(float(churn_prob))
    
    if churn_prob > 0.5:
        st.error(f"RỦI RO CAO: Xác suất rời bỏ là **{churn_prob:.1%}**")
        st.markdown(f"> **Gợi ý từ Tech Lead:** Khách hàng này có hợp đồng **{contract}** và cước phí **${monthly_charges}**. Hãy gọi ngay để tặng voucher giảm giá cước hoặc nâng cấp lên gói hợp đồng dài hạn!")
    elif churn_prob > 0.3:
        st.warning(f"RỦI RO TIỀM ẨN: Xác suất rời bỏ là **{churn_prob:.1%}**")
        st.markdown("> **Gợi ý:** Gửi email chăm sóc khách hàng và nhắc nhở về các lợi ích của dịch vụ.")
    else:
        st.success(f"KHÁCH HÀNG TRUNG THÀNH: Xác suất rời bỏ chỉ **{churn_prob:.1%}**")
        st.markdown("> **Gợi ý:** Rất an toàn. Có thể thực hiện chiến dịch bán chéo (Cross-sell) dịch vụ mới.")