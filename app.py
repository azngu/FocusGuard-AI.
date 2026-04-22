import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="FocusGuard AI", page_icon="🛡️")

st.title("🛡️ FocusGuard AI")
st.write("Dự đoán nhóm hành vi và nhận lời khuyên từ chuyên gia AI.")

# Nạp mô hình
if os.path.exists('focus_model.pkl'):
    model = joblib.load('focus_model.pkl')
    
    # Nhập liệu
    st.subheader("📝 Nhập thông số của bạn")
    st_val = st.slider("Tổng Screen Time (phút):", 0, 700, 250)
    pk_val = st.number_input("Số lần mở máy (Pickups):", 0, 200, 50)
    ft_val = st.slider("Thời gian tập trung (phút):", 0, 500, 180)

    if st.button("🚀 PHÂN TÍCH"):
        input_data = np.array([[st_val, pk_val, ft_val]])
        cluster = model.predict(input_data)[0]
        
        # Hệ thống chuyên gia mapping
        groups = {
            0: ("Deep Flow", "🧘‍♂️", "green", "Chiến binh Tập trung", "Duy trì kỷ luật tốt!", "Duy trì Pomodoro 50/10."),
            1: ("Smart Pulse", "✨", "blue", "Người dùng Thông thái", "Hiệu suất rất tốt!", "Tắt thông báo rác."),
            2: ("Steady Mode", "⚖️", "orange", "Trạng thái Thăng bằng", "Cần thêm sự ổn định.", "Đặt giới hạn 30p cho MXH."),
            3: ("Wandering Mind", "⚠️", "orange", "Tâm trí Lang thang", "Cảnh báo xao nhãng!", "Tập thở 4-7-8 trong 2 phút."),
            4: ("Digital Fog", "🚨", "red", "Màn sương Kỹ thuật số", "Báo động đỏ! Rời màn hình ngay.", "Nhìn xa 20m trong 20 giây.")
        }
        
        name, emoji, color, status, advice, action = groups[cluster]
        
        st.divider()
        st.markdown(f"### Nhóm: :{color}[{name} {emoji}]")
        st.info(f"**Lời khuyên:** {advice}")
        st.success(f"🎯 **Hành động:** {action}")
else:
    st.error("❌ Thiếu file 'focus_model.pkl'! Hãy chạy file train.py trước.")