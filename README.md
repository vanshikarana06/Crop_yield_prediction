# 🌾 Crop Yield Prediction System

## 📌 Overview
This project predicts agricultural crop yield using Machine Learning techniques. It was initially developed using Streamlit for rapid prototyping and later upgraded to Flask for better scalability and backend control.

---

## 🚀 Features
- Predict crop yield based on input parameters  
- Machine Learning model integration  
- Web-based interface  
- Two implementations:
  - Streamlit (prototype)
  - Flask (production-ready)

---

## 🛠️ Tech Stack
- Python  
- Flask (Backend)  
- Streamlit (Prototype UI)  
- Machine Learning (scikit-learn)  
- HTML, CSS (Frontend)  

---

## 📂 Project Structure

crop_yield_prediction/
│
├── flask/
│ ├── app.py
│ ├── templates/
│ └── static/
│
├── streamlit/
│ └── appmy.py
│
├── models/ # (model not uploaded due to size)
├── artifacts/
├── README.md


---

## 🤖 Machine Learning Model
- Model used: Random Forest  
- Trained on agricultural dataset  
- Includes preprocessing steps (encoding, scaling, transformations)

---

## ⚠️ Model File Note
Due to GitHub file size limitations:
> The trained model (`.joblib`) is not included in this repository.

👉 You can:
- Download it from external storage (Google Drive, etc.)
- Or retrain the model using the provided code

---

## ▶️ How to Run

### 🔹 Run Flask App
```bash
cd flask
python app.py
