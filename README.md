# 🌾 Crop Yield Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-red)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)

---

## 📌 Overview
An end-to-end Machine Learning system for predicting agricultural crop yields using environmental and soil parameters. The project follows a **dual architecture approach**:
- **Streamlit** for rapid prototyping and validation  
- **Flask** for production-ready backend deployment  

---

## 🚀 Features
- **Accurate Predictions:** Utilizes a Random Forest Regressor for reliable yield estimation  
- **Dual Deployment:** Streamlit prototype + Flask production backend  
- **Data Pipeline:** Feature engineering with encoding, scaling, and transformations  
- **Web Interface:** Custom HTML/CSS frontend integrated with backend  

---

## 🛠️ Tech Stack

| Category | Tools & Technologies |
|----------|--------------------|
| **Languages** | Python, HTML5, CSS3 |
| **Backend** | Flask (Production), Streamlit (Prototype) |
| **Machine Learning** | Scikit-learn, NumPy, Pandas |
| **Deployment** | Joblib, GitHub |

---

## 📂 Project Structure

```text
crop_yield_prediction/
├── flask/                # Production-ready Flask app
│   ├── app.py            # Backend logic
│   ├── templates/        # HTML frontend
│   └── static/           # CSS & assets
├── streamlit/            # Rapid prototyping
│   └── appmy.py          # Streamlit UI
├── models/               # Trained model files
├── artifacts/            # Scalers & encoders
└── README.md             # Documentation


## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Regressor *(robust to outliers in agricultural data)*  
- **Data Processing:** Standard scaling and categorical encoding  
- **Performance:** $R^2 \approx 0.89$  
- **Evaluation Metrics:** RMSE and R²  

---

## ⚠️ Model File

> **Note:**  
> Due to GitHub file size limitations, the trained `.joblib` model is not included.

### To run the project:
- Download the pre-trained model from external storage (e.g., Google Drive), or  
- Retrain the model using the provided dataset and training scripts  

---

## ▶️ How to Run

### 🔹 Option 1: Flask App (Production)
```bash
cd flask
python app.py
