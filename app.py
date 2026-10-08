import streamlit as st
import torch
import torch.nn as nn
import numpy as np

st.set_page_config(page_title="เบาก็หวาน งานก็เยอะ🥶", layout="centered")


class DiabetesNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 5),  # W1 5xnodes and bias1
            nn.ReLU(),  # Activation function
            nn.Linear(5, 1),  # W2 nodesx1 and bias2
            nn.Sigmoid(),  # result 0 - 1
        )

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_trained_model():
    checkpoint = torch.load("backend/diabetes_model(5x5).pth", weights_only=False)
    model = DiabetesNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # turn model into evaluation mode
    return model, checkpoint["mean"], checkpoint["std"], checkpoint["features"]


model, mean, std, features = load_trained_model()

st.title("เบาหวาน.AI.Here ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧")
st.caption(f"จริงจังไม่จริงโจ้ สั่งโกโก้ ได้ โอวัลติน \n\n หนักไม่เอา เบาไม่หวาน")

sample_cases = {
    "มีข้อมูลอยู่แล้ว? เอาเลยยย (Custom Input)": None,
    "1: Row 1 High risk case (Glucose 148, BloodPressure 72, BMI 33.6, Age 50)": [
        148.0,
        72.0,
        33.6,
        50.0,
        0.627,
    ],
    "2: Row 3 High risk case (Glucose 183, BloodPressure 64, BMI 23.3, Age 32)": [
        183.0,
        64.0,
        23.3,
        32.0,
        0.672,
    ],
    "3: Row 5 High risk case (Glucose 137, BloodPressure 40, BMI 43.1, Age 33)": [
        137.0,
        40.0,
        43.1,
        33.0,
        2.288,
    ],
    "4: Row 9 High risk case (Glucose 197, BloodPressure 70, BMI 30.5, Age 53)": [
        197.0,
        70.0,
        30.5,
        53.0,
        0.158,
    ],
    "5: Row 7 risk case (Glucose 78, BloodPressure 50, BMI 31.0, Age 26) False Negative": [
        78.0,
        50.0,
        31.0,
        26.0,
        0.248,
    ],
    "6: Row 2 Normal case (Glucose 85, BloodPressure 66, BMI 26.6, Age 31)": [
        85.0,
        66.0,
        26.6,
        31.0,
        0.351,
    ],
    "7: Row 4 Normal case (Glucose 89, BloodPressure 66, BMI 28.1, Age 21)": [
        89.0,
        66.0,
        28.1,
        21.0,
        0.167,
    ],
    "8: Row 6 Normal case (Glucose 116, BloodPressure 74, BMI 25.6, Age 30)": [
        116.0,
        74.0,
        25.6,
        30.0,
        0.201,
    ],
    "9: Row 8 Normal case (Glucose 115, BloodPressure 70, BMI 35.3, Age 29)": [
        115.0,
        70.0,
        35.3,
        29.0,
        0.134,
    ],
    "10: Row 11 Normal case (Glucose 110, BloodPressure 92, BMI 37.6, Age 30)": [
        110.0,
        92.0,
        37.6,
        30.0,
        0.191,
    ],
}

selected_case = st.selectbox("เลือกเคสตัวอย่าง(10 cases): ", list(sample_cases.keys()))

if sample_cases[selected_case] != None:
    values = sample_cases[selected_case]
    default_glucose, default_bp, default_bmi, default_age, default_dpf = (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
    )
else:
    default_glucose, default_bp, default_bmi, default_age, default_dpf = (
        120,
        70,
        25,
        25,
        0.35,
    )

dpf_case = {
    "custom": None,
    "ไม่มีคนในครอบครัวเป็นเบาหวานเลย": 0.15,
    "มีญาติห่างๆ เป็น (เช่น ลุง ป้า น้า อา)": 0.35,
    "มีคนในครอบครัวสายตรงเป็น (พ่อ แม่ หรือพี่น้อง)": 0.65,
    "มีทั้งพ่อ แม่ และญาติสายตรงหลายคนเป็น": 1.20,
}

st.subheader("ใส่ข้อมูลสุขภาพ")
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input(
            "ระดับน้ำตาลในเลือด (Glucose, mg/dL):",
            min_value=50.0,
            max_value=300.0,
            value=float(default_glucose),
        )
        bp = st.number_input(
            "ความดันโลหิต (Blood Pressure, mmHg):",
            min_value=30.0,
            max_value=200.0,
            value=float(default_bp),
        )
        bmi = st.number_input(
            "ดัชนีมวลกาย (BMI, kg/m²):",
            min_value=10.0,
            max_value=70.0,
            value=float(default_bmi),
        )

    with col2:
        age = st.number_input(
            # "อายุ (Age, ปี):", min_value=1.0, max_value=120.0, value=float(default_age)
            "อายุ (Age, ปี):", min_value=1, max_value=120, value=int(default_age)
        )
        dpf_title = "ประวัติกรรมพันธุ์ (Diabetes Pedigree Function):"
        if sample_cases[selected_case] == None:
            dpf_selected = st.selectbox(dpf_title,list(dpf_case.keys()))
            if dpf_case[dpf_selected] != None:
                default_dpf = dpf_case[dpf_selected]
                dpf_title = "ค่าประวัติกรรมพันธุ์"
            else:dpf_title = "ใส่ค่าประวัติกรรมพันธุ์ได้เลย"
        
        dpf = st.number_input(
            dpf_title,
            min_value=0.01,
            max_value=3.0,
            value=float(default_dpf),
            format="%.3f",
        )
        st.caption("ค่าเฉลี่ยคนทั่วไปอยู่ที่ประมาณ 0.35 - 0.50")

    # prediction button
    if st.button(
        "ประเมินความเสี่ยง (Run Prediction)", type="primary", use_container_width=True
    ):
        # Input Vector X (1 x 5)
        input_vector = torch.tensor([[glucose, bp, bmi, age, dpf]], dtype=torch.float32)

        # Standardization
        normalized_vector = (input_vector - mean) / std

        # Sent to Model
        with torch.no_grad():
            z1 = model.model[0](normalized_vector)
            prediction = model(normalized_vector)
            risk_prob = prediction.item() * 100  # scale to percent

        st.divider()

        # display
        st.subheader("ผลการประเมินความเสี่ยง")
        if risk_prob >= 50.0:
            st.error(f"**มีความเสี่ยงเป็นโรคเบาหวาน: {risk_prob:.2f}%**")
            st.write(
                "คำแนะนำ: ควรปรึกษาแพทย์เพื่อตรวจระดับน้ำตาลอย่างละเอียด และปรับพฤติกรรมการรับประทานอาหาร"
            )
        else:
            st.success(f"**ความเสี่ยงต่ำ (อยู่ในเกณฑ์ปกติ): {risk_prob:.2f}%**")
            st.write("คำแนะนำ: รักษาสุขภาพและออกกำลังกายอย่างสม่ำเสมอนะจ้ะะะะะ")

        # 5. show calculate Linear Algebra
        with st.expander("ดูเบื้องหลังการคำนวณทางคณิตศาสตร์ (Linear Algebra Detail)"):
            st.markdown(
                "**1. Input Vector (เวกเตอร์ข้อมูลดิบ) $\\mathbf{x} \\in \\mathbb{R}^{1 \\times 5}$:**"
            )
            st.write(input_vector.numpy())

            st.markdown(
                "**2. Normalized Vector (หลังลบ $\\boldsymbol{\\mu}$ หาร $\\boldsymbol{\\sigma}$):**"
            )
            st.write(normalized_vector.numpy())

            st.markdown(
                "**3. Weight Matrix $W_1$ ขนาด $5 \\times 5$ ค่า $b_1$ และ $z_1$ (Layer 1):**"
            )
            st.write("$W_1$",model.model[0].weight.data.numpy())
            st.write("$b_1$",model.model[0].bias.data.numpy())
            st.write("$z_1$",z1.numpy())

            st.markdown(
                "**4. Output Layer $W_2$ ขนาด $1 \\times 5$ ค่า $b_2$ และค่า Sigmoid:**"
            )
            st.write("$W_2$",model.model[2].weight.data.numpy())
            st.write("$b_2$",model.model[2].bias.data.numpy())
            st.write(
                f"ผลลัพธ์ก่อนเข้า Sigmoid ($z_2$) = {torch.logit(prediction).item():.4f}"
            )
            st.write(
                "ผลลัพธ์หลัง Sigmoid ($\hat{y} = \sigma(z_2)$) =",
                f"{prediction.item():.4f}",
            )
