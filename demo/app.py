import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Dự đoán sinh viên có học lực yếu kém")

col1, col2 = st.columns(2)
with col1:
    gioitinh = st.selectbox("Giới tính", ("Nam", "Nữ"))
with col2:
    khoahoc = st.selectbox("Khóa học", (8, 9, 10, 11, 12, 13, 14))

col3, col4 = st.columns(2)
with col3:
    hocky = st.number_input("Học kỳ", step=1, min_value=2 )
with col4:
    sotchk = st.number_input("Số tín chỉ học kỳ", step=1, min_value=0)

col5, col6 = st.columns(2)
with col5:
    dtbtl = st.number_input("Điểm trung bình các kì đã học")
with col6:
    drltl = st.number_input("Điểm rèn luyện các kì đã học", step=1)

col7, col8 = st.columns(2)
with col7:
    dtbhk_truoc = st.number_input("Điểm trung bình học kì trước")
with col8:
    drlhk_truoc = st.number_input("Điểm rèn luyện học kì trước", step=1)

somon_khongdat_hktruoc = st.number_input("Số môn không đạt học kì trước", step=1)

# 
button_style = '''
    <style>
        .stButton button {
            background-color: #0072B1;
            color: white;
            border-radius: 5px;
            font-weight: bold;
            padding: 8px 16px;
            box-shadow: none;
        }
        .stButton button:hover {
            color: white;
            background-color: #0072B1;
            box-shadow: none;
            border: none;
        }
    </style'''

st.markdown(button_style, unsafe_allow_html=True)

# If button is pressed
if st.button("Dự đoán"):
    
    # Unpickle classifier
    clf = joblib.load("clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[gioitinh, khoahoc, hocky, sotchk, dtbtl, drltl, dtbhk_truoc, drlhk_truoc, somon_khongdat_hktruoc]], 
                     columns = ["gioitinh", "khoahoc", "hocky_sx", "sotchk", "dtbtl", "drltl", "dtbhk_truoc", "drlhk_truoc", "somon_khongdat_hktruoc"])
    X = X.replace(["Nam", "Nữ"], [1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"Kết quả dự đoán: {prediction}")