# Customer Churn Prediction

A simple end-to-end ML project to predict customer churn and serve predictions via a Flask web app.

## Project Structure
- `model.py` — trains Logistic Regression, Decision Tree, and Random Forest; saves the best model to `models/churn_model.pkl`.
- `app.py` — Flask app that loads the saved model and serves a small form for predictions.
- `data/churn_data.csv` — sample dataset with three numeric features and a binary target `Churn`.
- `templates/` & `static/` — minimal UI files.

## Setup
```bash
pip install -r requirements.txt
python model.py          # trains and saves the model
python app.py            # runs the Flask app at http://127.0.0.1:5000
```

## Notes
- Form expects **three inputs**: tenure, monthly_charges, total_charges (in this order).
- Replace the sample dataset with your own for better performance.
