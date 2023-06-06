import streamlit as st
import pandas as pd
import joblib

# Hàm phân loại điểm rèn luyện
def convert_DRL_to_xeploai(x):
  if x < 35:
    return 0
  elif x >= 35 and x < 50:
    return 1
  elif x >= 50 and x < 65:
    return 2
  elif x >= 65 and x < 80:
    return 3
  elif x >= 80 and x < 90:
    return 4
  elif x >= 90:
    return 5

# Title
st.header("Dự đoán sinh viên có học lực yếu kém")

col1, col2 = st.columns(2)
with col1:
    gioitinh = st.selectbox("Giới tính", ("Nam", "Nữ"))
with col2:
    khoahoc = st.selectbox("Khóa học", (12, 13, 14, 15, 16, 17))

col3, col4 = st.columns(2)
with col3:
    hocky = st.number_input("Học kỳ", step=1, min_value=2)
with col4:
    sotchk = st.number_input("Số tín chỉ học kỳ", step=1, min_value=0)

col5, col6 = st.columns(2)
with col5:
    dtbtl = st.number_input("Điểm trung bình các kì đã học")
with col6:
    drltl = convert_DRL_to_xeploai(st.number_input("Điểm rèn luyện các kì đã học", step=1))

col7, col8 = st.columns(2)
with col7:
    dtbhk_truoc = st.number_input("Điểm trung bình học kì trước")
with col8:
    drlhk_truoc = convert_DRL_to_xeploai(st.number_input("Điểm rèn luyện học kì trước", step=1))

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
    </style>'''

st.markdown(button_style, unsafe_allow_html=True)

# If button is pressed
if st.button("Dự đoán"):
    # Unpickle classifier
    clf = joblib.load("./demo/clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame(
        [[gioitinh, khoahoc, hocky, sotchk , dtbtl, drltl, dtbhk_truoc, drlhk_truoc, somon_khongdat_hktruoc]],
        columns=[
            "gioitinh",
            "khoahoc",
            "hocky_sx",
            "sotchk",
            "dtbtl",
            "drltl",
            "dtbhk_truoc",
            "drlhk_truoc",
            "somon_khongdat_hktruoc",
        ],
    )
    X = X.replace(["Nam", "Nữ"], [1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    predict = "Không" if prediction == 1 else "Có"
    st.text(f"Có nguy cơ học lực yếu, kém ở kỳ tiếp theo hay không?: {predict}")

# Batch prediction
st.header("Dự đoán hàng loạt")
uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

if uploaded_file is not None:
    # Load classifier
    clf = joblib.load("./demo/clf.pkl")

    df = pd.read_csv(uploaded_file)
    
    # Preprocess input data
    # df = df.replace(["Nam", "Nữ"], [1, 0])
    
    # Make predictions
    predictions = clf.predict(df)

    # Add predictions as a new column in the DataFrame
    df['predict'] = predictions

    # # Output input data and predictions
    # st.dataframe(df)


    # Export DataFrame to CSV
    # df.to_csv('output.csv', index=False)
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)
    


    # Count the labels
    label_counts = df['predict'].value_counts()

    # Output label counts
    if len(label_counts) == 1:
        label = label_counts.index[0]
        if label == 0:
            st.write(f"Tất cả sinh viên đều rớt")
        else:
            st.write(f"Tất cả sinh viên đều đậu")
    else:
        st.write("Số sinh viên dự đoán học lực yếu kém:", label_counts[0])
        st.write("Số sinh viên dự đoán học lực trên trung bình:", label_counts[1])
        st.download_button(
           
    "Press to Download cvs file",
    csv,
    "predicted.csv",
    "text/csv",
    key='download-csv'
    )

