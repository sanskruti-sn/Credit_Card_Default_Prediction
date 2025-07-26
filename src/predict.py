import numpy as np

def predict_default(model, scaler, input_data):
    """
    input_data: list or np.array of 23 features
    """
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return prediction[0]
