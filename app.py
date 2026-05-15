import streamlit as st
import numpy as np
import joblib
import os

# Cấu hình trang
st.set_page_config(
    page_title="FocusGuard AI - Trình Bảo Vệ Tập Trung", 
    page_icon="🛡️",
    layout="centered"
)

# Tùy chỉnh giao diện bằng CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 15px;
    }
    .intro-box {
        background-color: #eef2ff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4f46e5;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề và Giới thiệu
st.title("🛡️ FocusGuard AI")

st.markdown("""
    <div class="intro-box">
        <h4 style="color: #1e1b4b; margin-top: 0;">Hệ thống phân tích thông minh</h4>
        <p style="color: #3730a3; font-size: 0.95rem; margin-bottom: 0;">
            Ứng dụng sử dụng thuật toán máy học để đánh giá trạng thái tâm lý dựa trên 
            <b>thời gian sử dụng máy</b>, <b>tần suất mở máy (pickups)</b> và <b>thời gian tập trung thực tế</b>. 
            Từ đó, AI sẽ xác định nhóm hành vi của bạn và đưa ra lộ trình cải thiện sự tập trung cá nhân hóa.
        </p>
    </div>
""", unsafe_allow_html=True)

# Kiểm tra file mô hình
if os.path.exists('focus_model.pkl'):
    model = joblib.load('focus_model.pkl')
    
    # Khu vực nhập liệu
    st.subheader("📝 Thông số hoạt động của bạn")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st_val = st.slider("Tổng Screen Time (phút):", 0, 700, 250, help="Tổng thời gian bạn nhìn vào màn hình điện thoại/máy tính.")
        ft_val = st.slider("Thời gian tập trung (phút):", 0, 500, 180, help="Thời gian bạn thực sự làm việc sâu mà không bị ngắt quãng.")
    
    with col2:
        pk_val = st.number_input("Số lần mở máy:", 0, 200, 50, help="Số lần bạn cầm điện thoại lên và mở khóa.")

    st.markdown("---")

    if st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True):
        input_data = np.array([[st_val, pk_val, ft_val]])
        cluster = model.predict(input_data)[0]
        
        # Hệ thống chuyên gia mapping
        groups = {
            0: {
                "name": "Deep Flow", "emoji": "🧘‍♂️", "color": "green",
                "advice": "Bạn đang sở hữu khả năng tập trung tuyệt vời. Hãy duy trì nhịp độ này để đạt hiệu suất tối đa.",
                "action": "Duy trì chu kỳ làm việc sâu (Deep Work) 90 phút và nghỉ ngơi 15 phút."
            },
            1: {
                "name": "Smart Pulse", "emoji": "✨", "color": "blue",
                "advice": "Cách bạn sử dụng thiết bị khá thông minh và cân bằng.",
                "action": "Kiểm tra và tắt các thông báo không quan trọng từ ứng dụng mạng xã hội."
            },
            2: {
                "name": "Steady Mode", "emoji": "⚖️", "color": "orange",
                "advice": "Sự tập trung của bạn ở mức trung bình, nhưng có dấu hiệu bị xao nhãng bởi các thông báo.",
                "action": "Sử dụng tính năng 'Screen Time' để giới hạn 30 phút mỗi ngày cho các app giải trí."
            },
            3: {
                "name": "Wandering Mind", "emoji": "⚠️", "color": "orange",
                "advice": "Tâm trí bạn đang có xu hướng nhảy vọt giữa các tác vụ. Số lần mở máy quá cao làm gián đoạn dòng chảy tư duy.",
                "action": "Áp dụng bài tập thở 4-7-8 trong 2 phút mỗi khi có ý định cầm điện thoại vô thức."
            },
            4: {
                "name": "Digital Fog", "emoji": "🚨", "color": "red",
                "advice": "Cảnh báo! Bạn đang rơi vào tình trạng quá tải kỹ thuật số. Não bộ cần được nghỉ ngơi ngay lập tức.",
                "action": "Thực hiện quy tắc 20-20-20: Nhìn xa 20 feet (6m) trong 20 giây sau mỗi 20 phút làm việc."
            }
        }
        
        res = groups[cluster]
        
        # Hiển thị kết quả
        st.markdown(f"### Nhóm: :{res['color']}[{res['name']} {res['emoji']}]")
        
        with st.container():
            st.info(f"💡 **Lời khuyên chuyên gia:** {res['advice']}")
            st.success(f"🎯 **Hành động cụ thể:** {res['action']}")

else:
    st.error("❌ Không tìm thấy file 'focus_model.pkl'! Vui lòng đảm bảo bạn đã chạy mã huấn luyện mô hình trước.")

# Footer
st.markdown("---")
st.caption("© 2026 FocusGuard AI - Công cụ hỗ trợ kỷ luật số cá nhân.")
