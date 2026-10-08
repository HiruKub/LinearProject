# 🩺 ระบบประเมินความเสี่ยงโรคเบาหวานเบื้องต้นด้วยโครงข่ายประสาทเทียม
### (Diabetes Risk Classification using Simple Neural Network)
> **โครงงานวิชาคณิตศาสตร์วิศวกรรม: พีชคณิตเชิงเส้น (Linear Algebra Project)**  
> สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง (KMITL)

---

## 📌 1. บทนำและที่มาของโครงงาน (Introduction)
โรคเบาหวานเป็นโรคไม่ติดต่อเรื้อรัง (NCDs) ที่ส่งผลกระทบต่อสุขภาพในระยะยาวและอาจก่อให้เกิดภาวะแทรกซ้อนรุนแรง การคัดกรองความเสี่ยงในระยะเริ่มต้นช่วยให้บุคคลสามารถปรับเปลี่ยนพฤติกรรมการใช้ชีวิตหรือเข้ารับการวินิจฉัยจากแพทย์ได้อย่างทันท่วงที 

โครงงานนี้พัฒนาแอพพลิเคชันจำแนกประเภท (Classification) เพื่อประเมินความเสี่ยงโรคเบาหวาน โดยประยุกต์ใช้ **ทฤษฎีเวกเตอร์และเมทริกซ์ (Vector & Matrix Theory)** ร่วมกับ **โครงข่ายประสาทเทียมอย่างง่าย (Simple Multi-Layer Perceptron)** เพื่อให้เห็นขั้นตอนการประมวลผลและการคำนวณทางพีชคณิตเชิงเส้นอย่างเป็นรูปธรรมตั้งแต่ต้นจนได้ผลลัพธ์

---

## 📊 2. แหล่งข้อมูล (Dataset & Preprocessing)
* **แหล่งข้อมูล:** [Pima Indians Diabetes Database (Kaggle / UCI Machine Learning Repository)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **จำนวนข้อมูล:** 768 เวกเตอร์ข้อมูล (ผ่านเกณฑ์ขั้นต่ำ 150 ชุดข้อมูล)
* **ฟีเจอร์ที่นำมาใช้ (5 Features):**
  1. `Glucose`: ระดับน้ำตาลในเลือด (mg/dL)
  2. `BloodPressure`: ความดันโลหิตไดแอสโตลิก (mmHg)
  3. `BMI`: ดัชนีมวลกาย (kg/m²)
  4. `Age`: อายุ (ปี)
  5. `DiabetesPedigreeFunction`: คะแนนประวัติกรรมพันธุ์โรคเบาหวานในครอบครัว
* **การเตรียมและทำความสะอาดข้อมูล (Data Cleaning):**
  - ตรวจสอบและจัดการค่าสูญหายที่ถูกบันทึกเป็น 0 ที่ผิดธรรมชาติทางการแพทย์ ในคอลัมน์ `Glucose`, `BloodPressure`, และ `BMI` โดยแทนที่ด้วยค่ามัธยฐาน (Median Imputation)
  - สุ่มสลับและแบ่งชุดข้อมูลด้วย `torch.randperm` (Seed: 42) ออกเป็น:
    - **Train Set (80%):** 614 เวกเตอร์ข้อมูล
    - **Test Set (20%):** 154 เวกเตอร์ข้อมูล

---

## 📐 3. ทฤษฎีพีชคณิตเชิงเส้นที่ประยุกต์ใช้ (Linear Algebra Application)

### 3.1 ทฤษฎีเวกเตอร์ (Vector Theory)
1. **Input Vector ($\mathbf{x} \in \mathbb{R}^{1 \times 5}$):** จัดเก็บข้อมูลสุขภาพของผู้ป่วยแต่ละรายให้อยู่ในรูปเวกเตอร์แถว
   $$\mathbf{x} = \begin{bmatrix} x_{\text{Glucose}} & x_{\text{BloodPressure}} & x_{\text{BMI}} & x_{\text{Age}} & x_{\text{DPF}} \end{bmatrix}$$
2. **Feature Standardization (Vector Element-wise Operation):**
   ปรับสเกลข้อมูลให้อยู่ในหน่วยมาตรฐานโดยใช้ Mean Vector ($\boldsymbol{\mu} \in \mathbb{R}^{1 \times 5}$) และ Standard Deviation Vector ($\boldsymbol{\sigma} \in \mathbb{R}^{1 \times 5}$) เพื่อป้องกันไม่ให้ฟีเจอร์ที่มีสเกลใหญ่ครอบงำการคำนวณ:
   $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$
3. **Linear Combination & Dot Product:** การหาผลรวมเชิงเส้นระหว่างเวกเตอร์คุณลักษณะกับเวกเตอร์น้ำหนักในแต่ละโหนดการคำนวณ

### 3.2 ทฤษฎีเมทริกซ์ (Matrix Theory)
1. **Weight Matrix ($W_1 \in \mathbb{R}^{5 \times 5}$):** เมทริกซ์น้ำหนักสำหรับ Hidden Layer ชั้นแรก (Square Matrix ขนาด $5 \times 5$)
2. **Matrix-Vector Multiplication (Forward Propagation):**
   * **Hidden Layer (Layer 1):**
     $$\mathbf{z}_1 = \mathbf{x}_{\text{norm}} W_1 + \mathbf{b}_1 \quad \in \mathbb{R}^{1 \times 5}$$
     $$\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1) = \max(0, \mathbf{z}_1)$$
   * **Output Layer (Layer 2):**
     $$z_2 = \mathbf{a}_1 W_2 + b_2 \quad \in \mathbb{R}^{1 \times 1} \quad (\text{โดย } W_2 \in \mathbb{R}^{5 \times 1}, b_2 \in \mathbb{R})$$
     $$\hat{y} = \sigma(z_2) = \frac{1}{1 + e^{-z_2}} \quad (\text{แปลงเป็นความน่าจะเป็นความเสี่ยง 0 - 1})$$

---

## 🧠 4. สถาปัตยกรรมโมเดลและผลการทดลอง (Model Architecture & Results)

* **โครงสร้างโมเดล:** Input (5) $\to$ Hidden Layer (5 nodes, ReLU) $\to$ Output (1 node, Sigmoid)
* **Optimization:** Loss Function: `Binary Cross Entropy (BCELoss)`, Optimizer: `Adam (lr=0.01, weight_decay=0.01)`, Epochs: `1000`
* **ผลการเปรียบเทียบสถาปัตยกรรม (Ablation Study):**
  | สถาปัตยกรรม | ขนาด $W_1$ | Train Loss | Test Accuracy | ผลการวิเคราะห์ |
  | :---: | :---: | :---: | :---: | :--- |
  | **Hidden 4 nodes** | $4 \times 4$ | 0.4477 | **76.62%** | มีความสมดุลสูง ไม่ Overfit และคำนวณมือง่ายสุด |
  | **Hidden 5 nodes** | $5 \times 5$ | 0.4476 | **75.97%** | รองรับ DPF ครบถ้วน ได้เมทริกซ์จัตุรัสสมมาตร |
  | **Hidden 6 nodes** | $5 \times 6$ | 0.4220 | 74.68% | Loss ลดลง แต่ความแม่นยำบน Test Set ลดลง |
  | **Hidden 8 nodes** | $5 \times 8$ | 0.3930 | 74.68% | Train Loss ต่ำสุดแต่เกิด Overfitting ชัดเจน |

---

## 📁 5. โครงสร้างโปรเจกต์ (Project Structure)
```text
LinearProject/
├── backend/
│   ├── clean_data.py               # สคริปต์ตรวจสอบและทำความสะอาดข้อมูล
│   ├── train.py                    # สคริปต์เทรนโมเดล วัดผล และบันทึก Weights
│   └── diabetes_model(5x5).pth     # Checkpoint โมเดลและค่า Mean/Std
├── data/
│   └── diabetes.csv                # ชุดข้อมูล Pima Indians Diabetes
├── app.py                          # เว็บแอพพลิเคชัน Interactive Demo (Streamlit)
├── pyproject.toml                  # ไฟล์จัดการ Dependencies และโปรเจกต์
└── README.md                       # เอกสารประกอบโครงงาน
```

---

## 💻 6. วิธีการติดตั้งและการใช้งาน (Installation & Usage)

### 6.1 การติดตั้งสภาพแวดล้อม (Environment Setup)
```powershell
# ติดตั้ง dependencies ผ่าน uv
uv sync
```

### 6.2 การเทรนโมเดล (Train Model)
```powershell
.venv\Scripts\python.exe backend/train.py
```

### 6.3 การรันเว็บแอพพลิเคชัน Demo (Streamlit)
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
*(เข้าใช้งานผ่านเว็บเบราว์เซอร์ที่ `http://localhost:8501` มีระบบเลือก 10 เคสตัวอย่างให้ทดสอบ พร้อมแสดงขั้นตอนการคำนวณเมทริกซ์และเวกเตอร์แบบเรียลไทม์)*

---

## 👥 7. สมาชิกผู้จัดทำ (Contributors)
* **68011025** นายวิชาธร ขจรเนติกุล
* **68011812** นางสาวศรัณย์พร หนองช้าง
* **68011848** นายอัษฎาวัชร สามสีเนียม

> *จัดทำขึ้นเพื่อประกอบการเรียนวิชาคณิตศาสตร์วิศวกรรม: พีชคณิตเชิงเส้น (Linear Algebra)*  
> *คณะวิศวกรรมศาสตร์ สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง*
