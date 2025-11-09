# Crop-Recommendation-App
Streamlit-based Crop Recommendation System using ML
# 🌾 Intelligent Crop Recommendation System

This project is a **Machine Learning + Streamlit** based web application that recommends the most suitable crop to grow based on soil and climatic conditions.

---

## 🚀 Project Overview
Farmers often face challenges in selecting the right crop for their soil and climate.  
This **AI-driven solution** uses data on nutrients, temperature, humidity, rainfall, and pH to predict the best crop using advanced ML algorithms like **Random Forest, SVM, and XGBoost**.

---

## 🧠 Tech Stack
- **Programming Language:** Python  
- **Framework:** Streamlit  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Model Storage:** Pickle  

---

## 📂 Dataset
The dataset used is the **Crop Recommendation Dataset**, containing soil and weather parameters such as:
- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- Temperature (°C)
- Humidity (%)
- pH
- Rainfall (mm)
- Crop label (target)

---

## ⚙️ Model Workflow
1. **Data Cleaning & Preprocessing**
2. **Feature Encoding (LabelEncoder)**
3. **Feature Scaling (StandardScaler)**
4. **Model Training (SVM, Random Forest, XGBoost)**
5. **Model Evaluation (Accuracy, R² Score)**
6. **Saving Trained Models (`.pkl` files)**
7. **Deploying on Streamlit**

---

## 💻 How to Run the App

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Lokeshwaran2109/Crop-Recommendation-App.git
cd Crop-Recommendation-App
