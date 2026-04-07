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


models_dir = os.path.join(project_root, 'models')
train_processed_path = os.path.join(project_root, 'data', 'processed', 'train.csv')


sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.dpi"] = 120


## Mở file train dataset đã được xử lý nằm trong folder data/processed
try:
    df_train = pd.read_csv(train_processed_path)
    print("Mở file train dataset thành công !!")
except FileNotFoundError:
    print(f"Không tìm thấy file tại đường dẫn {train_processed_path}. Vui lòng kiểm tra lại !!")
    exit()


X_train_scaled = df_train.drop(columns=['Churn'])
y_train = df_train['Churn']


lr_val_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}

grid_lr = GridSearchCV(estimator=lr_val_model, param_grid=param_grid_lr, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_lr.fit(X_train_scaled, y_train)
best_lr = grid_lr.best_estimator_
print(f"Tham số tốt nhất cho Logistic Regression: {grid_lr.best_params_}")


dt_val_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf': [5, 10, 20]
}

grid_dt = GridSearchCV(estimator=dt_val_model, param_grid=param_grid_dt, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_dt.fit(X_train_scaled, y_train)
best_dt = grid_dt.best_estimator_
print(f"Tham số tốt nhất cho Decision Tree: {grid_dt.best_params_}")

joblib.dump(best_lr, os.path.join(models_dir, 'best_logistic_model.pkl'))
joblib.dump(best_dt, os.path.join(models_dir, 'best_tree_model.pkl'))

print("HUẤN LUYỆN MÔ HÌNH THÀNH CÔNG !!")
