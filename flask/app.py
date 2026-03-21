from flask import Flask, render_template, request
from ml_handler import preprocess_and_predict, artifacts

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html', cat_options=artifacts["cat_levels"])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate required fields 
        required_num = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop_Year']
        required_cat = artifacts["cat_cols"]   
        missing = [k for k in required_num + required_cat if not request.form.get(k, '').strip()]
        if missing:
            return render_template(**'home.html', error=f'Missing fields: {", ".join(missing)}', cat_options=artifacts["cat_levels"])
        
        # Get data
        data = {
            "Area": float(request.form['Area']),
            "Annual_Rainfall": float(request.form['Annual_Rainfall']),
            "Fertilizer": float(request.form['Fertilizer']),
            "Pesticide": float(request.form['Pesticide']),
            "Crop_Year": int(request.form['Crop_Year']),
        }
        for col in artifacts["cat_cols"]:
            data[col] = request.form[col]

        prediction = preprocess_and_predict(data)
        total_prod = prediction * data["Area"]

        return render_template('home.html', 
                               prediction=round(prediction, 2), 
                               total=round(total_prod, 2),
                               cat_options=artifacts["cat_levels"])
    except ValueError as e:
        return render_template('home.html', error=f'Invalid number input: {str(e)}', cat_options=artifacts["cat_levels"])
    except Exception as e:
        return render_template('home.html', error=f'Prediction error: {str(e)}', cat_options=artifacts["cat_levels"])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

