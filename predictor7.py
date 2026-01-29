# app.py
import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. 加载模型（训练列顺序：SOFA,CRRT,SpO2,MBP,HR,Age,CAR）
model = joblib.load('RF.pkl')

# 2. 页面
st.set_page_config(page_title="ICU-Outcome预测", layout="centered")
st.title("ICU 结局（Outcome）预测器")
st.markdown("请录入以下 7 项指标：")

# 3. 输入组件
col1, col2 = st.columns(2)
with col1:
    sofa = st.number_input("SOFA 评分", value=6, step=1, help="常见范围 0–24")
    spo2 = st.number_input("SpO₂ (%)",  value=95, step=1)
    mbp  = st.number_input("MBP (mmHg)", value=80.0, step=1.0)
    hr   = st.number_input("HR (次/分)", value=90, step=1)
with col2:
    crrt = st.selectbox("是否 CRRT", [0,1], format_func=lambda x:"否" if x==0 else "是")
    age  = st.number_input("年龄", value=65, step=1)
    car  = st.number_input("CAR",  value=2.0, step=0.1)

# 4. 预测
if st.button("预测"):
    X = np.array([[sofa, crrt, spo2, mbp, hr, age, car]])
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0]

    st.subheader("Outcome 预测结果")
    label = "阳性（1）" if pred == 1 else "阴性（0）"
    st.write(f"**Outcome = {pred}（{label}）**")
    st.write(f"P(Outcome=0) = {prob[0]:.2%} | P(Outcome=1) = {prob[1]:.2%}")

    if pred == 1:
        st.warning("模型判断 Outcome=1，建议临床加强关注与干预。")
    else:
        st.success("模型判断 Outcome=0，维持当前管理即可。")