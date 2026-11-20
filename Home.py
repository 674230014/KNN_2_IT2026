from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกผู้เป็นโรคหัวใจ")
#st.image("./img/chitsanupong.jpg")
col1, col2, col3 = st.columns(3)

with col1:
   st.header("heartA")
   st.image("./img/heartA.jpg")

with col2:
   st.header("heartB")
   st.image("./img/heartB.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

dt1 = dt['Age'].mean()
dt2 = dt['Sex'].mean()
dt3 = dt['ChestPainType'].mean()
dt4 = dt['MaxHR'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
   st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")


Age = st.slider("กรุณาเลือกข้อมูล Age")
Sex = st.slider("กรุณาเลือกข้อมูล Sex")
ChestPainType = st.slider("กรุณาเลือกข้อมูล Chest Pain Type")
MaxHR = st.slider("กรุณาเลือกข้อมูล Max Heart Rate")

Age = st.number_input("กรุณาป้อนข้อมูล Age (อายุ)", min_value=0, max_value=120, value=50)
Sex = st.number_input("กรุณาป้อนข้อมูล Sex (เพศ 0 หรือ 1)", min_value=0, max_value=1, value=1)
ChestPainType = st.number_input("กรุณาป้อนข้อมูล Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
MaxHR = st.number_input("กรุณาป้อนข้อมูล Max Heart Rate", min_value=60, max_value=220, value=150)

if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris.csv") 

   features = ['Age', 'Sex', 'ChestPainType', 'MaxHR']
   X = dt[features]
   y = dt['HeartDisease']
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)

   x_input = pd.DataFrame([[Age, Sex, ChestPainType, MaxHR]], columns=features) #ข้อมูลใหม่สำหรับทำนาย

   out = Knn_model.predict(x_input)
   st.write(out)

   if out[0] == 0:
    st.success("ผลการทำนาย: ไม่เป็นโรคหัวใจ")
    st.image("./img/heartA.jpg")
   else:
    st.error("ผลการทำนาย: เป็นโรคหัวใจ")
    st.image("./img/heartB.jpg")

else:
    st.write("ไม่ทำนาย")
