import pandas as pd
from google.cloud import storage
import joblib
from flask import jsonify, request

# Load the model from GCS-bucket
def load_model():
    storage_client = storage.Client()
    bucket = storage_client.bucket("mlops-assign2-bucket")
    blob = bucket.blob("tuned_model.pkl")
    blob.download_to_filename("/tmp/tuned_model.pkl")
    model = joblib.load("/tmp/tuned_model.pkl")
    return model

# Prediction function for Google Cloud Functions
def predict_function(request):
    try:
        # Parse the JSON request
        request_json = request.get_json()
        if 'data' not in request_json:
            return jsonify({"error": "Invalid input, 'data' field is missing"}), 400
        
        # Get the input data
        input_data = request_json['data']
        columnz = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Create a DataFrame with the input data
        df = pd.DataFrame([input_data], columns=columnz)
        
        # Load the model
        model = load_model()
        
        # Make the prediction
        prediction = model.predict(df)
        
        # Return the prediction
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500