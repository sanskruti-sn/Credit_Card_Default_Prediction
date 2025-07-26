from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load scaler and model
scaler = pickle.load(open('notebooks/notebooks/artifacts/scaler.pkl', 'rb'))
model = pickle.load(open('notebooks/notebooks/artifacts/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[f'feature{i}']) for i in range(23)]
        input_scaled = scaler.transform([input_data])
        pred = model.predict(input_scaled)[0]
        result = "Default" if pred == 1 else "No Default"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
