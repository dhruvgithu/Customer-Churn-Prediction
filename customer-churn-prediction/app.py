from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/churn_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tenure = float(request.form.get('tenure'))
        monthly_charges = float(request.form.get('monthly_charges'))
        total_charges = float(request.form.get('total_charges'))
        features_array = np.array([[tenure, monthly_charges, total_charges]])
        prediction = model.predict(features_array)[0]
        result = "Customer is likely to CHURN" if int(prediction) == 1 else "Customer will STAY"
        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
