# 🌾 Crop Yield Prediction App

This project builds a **Machine Learning model** to predict crop yield (tons/hectare) using agricultural and environmental features such as **Area, Rainfall, Fertilizer, Pesticide, Season, Crop, and Location**.  
It also includes a **Streamlit web app** for easy interaction and deployment.

---

## 📌 Features of the Project
- ✅ Data preprocessing (outlier capping, log transformation, categorical encoding)  
- ✅ Feature engineering (leakage column removal, consistent column ordering)  
- ✅ Random Forest Regressor with **hyperparameter tuning (RandomizedSearchCV)**  
- ✅ Model evaluation using **R², RMSE** and cross-validation  
- ✅ Export of preprocessing artifacts (`preprocessing.json`) for reproducibility  
- ✅ **Streamlit app** for interactive predictions  

---
## 🗂️ Project Structure
```
📦 crop-yield-prediction
│── crop_data.csv # Original dataset 
├── models/
│ └── best_model_rf.joblib # Trained Random Forest model
├── artifacts/
│ └── preprocessing.json # Encoded mappings, caps, feature order
│── streamlit_app.py # Streamlit app for deployment
│── Untitled1.ipynb
├── README.md # Project documentation
└── requirements.txt # Dependencies
```
## ⚙️ Installation & Setup

1. Clone the repository
   ```bash
   git clone https://github.com/vanshikarana06/crop-yield-prediction.git
   cd crop-yield-prediction
   ```
2. Create a virtual environment & install dependencies
   ```bash

    pip install -r requirements.txt
    Run Jupyter notebooks (for training / experiments)
   ```
 ```bash

Launch the Streamlit app
streamlit run app/streamlit_app.py
```
## 📊 Model Training Workflow
- Data Preprocessing

- Cap numeric features at 1st and 99th percentiles

- Apply log1p transform to skewed numeric features

- Encode categorical columns using consistent mappings

- Drop leakage features (e.g., Production)

- Model Training

- Random Forest Regressor

- Hyperparameter tuning with RandomizedSearchCV

- 5-fold cross-validation

## Evaluation

- R² ≈ 0.89 on original scale

- RMSE ≈ 285 tons/hectare

## 🚀 Streamlit App Features
- User-friendly sliders and dropdowns for input

- Instant yield prediction (tons/hectare)

- Total production calculated from area × yield

- Handles unseen categories with fallback encoding

## 📌 Example Prediction
Input:

- Area: 2.5 hectares

- Annual Rainfall: 1200 mm

- Fertilizer: 80 kg/hectare

- Pesticide: 10 kg/hectare

- Season: Kharif

- Crop: Rice

Output:

- ✅ Expected Yield: 3.42 tons/hectare  
- 📦 For your 2.5 hectares, total expected production is 8.55 tons.

 ## 🛠️ Tech Stack
- Python 3.9+

- Pandas, NumPy, Matplotlib, Seaborn

- Scikit-learn

- Streamlit

- Joblib, JSON

## 📌 Future Improvements
- Add more ML models (XGBoost, LightGBM) for comparison

- Deploy the app to Streamlit Cloud / Heroku / AWS

- Integrate real-time weather APIs for rainfall data

- Create a dashboard for farmers & policymakers

## 👨‍💻 Author
**Vanshika Rana**  
📧 [vanshurana1706@gmail.com]  
🌐 [LinkedIn](www.linkedin.com/in/vanshika-rana-143776327) | [GitHub](https://github.com/vanshikarana06)
