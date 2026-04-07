import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

## Tạo đường dẫn đến các thư mục
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

images_dir = os.path.join(project_root, 'images')
models_dir = os.path.join(project_root, 'models')
test_processed_path = os.path.join(project_root, 'data', 'processed', 'test.csv')


sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.dpi"] = 120

## Mở file train dataset đã được xử lý nằm trong folder data/processed
try:
    df_test = pd.read_csv(test_processed_path)
    print("Mở file test dataset thành công !!")
except FileNotFoundError:
    print(f"Không tìm thấy file tại đường dẫn {test_processed_path}. Vui lòng kiểm tra lại !!")
    exit()

X_test_scaled = df_test.drop(columns=['Churn'])
y_test = df_test['Churn']

lr_model_path = os.path.join(models_dir, 'best_logistic_model.pkl')
dt_model_path = os.path.join(models_dir, 'best_tree_model.pkl')

try:
    best_lr_model = joblib.load(lr_model_path)
    best_dt_model = joblib.load(dt_model_path)
    print("Load thành công 2 model !!")
except FileNotFoundError:
    print("Không tìm thấy file lưu mô hình !!")
    exit()

y_pred_lr = best_lr_model.predict(X_test_scaled)
y_pred_dt = best_dt_model.predict(X_test_scaled)

print("\nLOGISTIC REGRESSION")
print(classification_report(y_test, y_pred_lr))

print("\nDECISION TREE")
print(classification_report(y_test, y_pred_dt))


#------------------------------------------------
### VẼ VÀ LƯU BIỂU ĐỒ ĐÁNH GIÁ
#------------------------------------------------

## Vẽ ma trận nhầm lẫn
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_ylabel('Thực tế')
axes[0].set_xlabel('Dự đoán')

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[1].set_title('Decision Tree Confusion Matrix')
axes[1].set_ylabel('Thực tế')
axes[1].set_xlabel('Dự đoán')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '05_confusion_matrix.png'), bbox_inches='tight')
plt.close()

## Vẽ ROC Curve
y_prob_lr = best_lr_model.predict_proba(X_test_scaled)[:, 1]
y_prob_dt = best_dt_model.predict_proba(X_test_scaled)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc(fpr_lr, tpr_lr):.2f})', linewidth=2)
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc(fpr_dt, tpr_dt):.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(images_dir, '06_roc_curve.png'), bbox_inches='tight')
plt.close()

print("Đã vẽ thành công và lưu toàn bộ các biểu đồ vào thư mục 'images/'")
print("ĐÁNH GIÁ MÔ HÌNH THÀNH CÔNG !!")