import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib

# Tạo dữ liệu giả lập 50 mẫu để huấn luyện
np.random.seed(42)
data = {
    'Screen_Time': np.random.randint(50, 700, 50),
    'Pickups': np.random.randint(10, 200, 50),
    'Focus_Time': np.random.randint(20, 450, 50)
}
df = pd.DataFrame(data)

# Huấn luyện mô hình K-Means với 5 nhóm
X = df[['Screen_Time', 'Pickups', 'Focus_Time']]
model = KMeans(n_clusters=5, random_state=42, n_init=10)
model.fit(X)

# Xuất file bộ não .pkl
joblib.dump(model, 'focus_model.pkl')
print("✅ Đã tạo xong file 'focus_model.pkl'. Hãy giữ file này cùng thư mục với app.py!")