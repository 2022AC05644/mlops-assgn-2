from pycaret.classification import load_model, predict_model
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import joblib
import shap  # Import SHAP for explanations

app = Flask(__name__)

# Load the pre-trained model
model = load_model('best_diabetes_model')

# Load the SHAP explainer from the saved file
explainer = joblib.load('shap_explainer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        age = float(request.form['age'])
        insulin = float(request.form['insulin'])
        skin_thickness = float(request.form['skin_thickness'])
        bp = float(request.form['bp'])
        pedigree = float(request.form['pedigree'])
        pregnancies = float(request.form['pregnancies'])

        # Prepare input data in the same structure used during training
        input_data = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, pedigree, age]],
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                           'BMI', 'DiabetesPedigreeFunction', 'Age'])

        # Make prediction using PyCaret's predict_model
        prediction_result = predict_model(model, data=input_data)
        
        # Extract the prediction label
        prediction = prediction_result['prediction_label'][0]
        confidence = prediction_result['prediction_score'][0]*100  # Assuming 'Score' gives confidence

        # Calculate SHAP values for the input data
        shap_values = explainer(input_data)

        print(shap_values.values[0])
        print(input_data.columns)

        # Generate a textual explanation based on SHAP values
        # Generate a textual explanation based on SHAP values
        explanations = []

        for feature, shap_value in zip(input_data.columns, shap_values.values[0]):
            if feature == 'Glucose':
                explanations.append(f"Glucose Level (contributed {'positively' if shap_value > 0 else 'negatively'})")
            elif feature == 'BMI':
                explanations.append(f"Body Mass Index (BMI) (contributed {'positively' if shap_value > 0 else 'negatively'})")
            elif feature == 'Insulin':
                explanations.append(f"Insulin (contributed {'positively' if shap_value > 0 else 'negatively'})")
            elif feature == 'DiabetesPedigreeFunction':
                explanations.append(f"Diabetes Pedigree Function (contributed {'positively' if shap_value > 0 else 'negatively'})")
            elif feature == 'BloodPressure':
                explanations.append(f"Blood Pressure (contributed {'positively' if shap_value > 0 else 'negatively'})")
            elif feature == 'SkinThickness':
                explanations.append(f"Skin Thickness (contributed {'positively' if shap_value > 0 else 'negatively'})")
            elif feature == 'Age':
                explanations.append(f"Age (contributed {'positively' if shap_value > 0 else 'negatively'})")
            elif feature == 'Pregnancies':
                explanations.append(f"Pregnancies (contributed {'positively' if shap_value > 0 else 'negatively'})")

            # Join the explanations into a single string
        explanations_text = "<br>".join(explanations)

        # Display the prediction, confidence, and explanations
        result_text = f"{'Diabetic' if prediction == 1 else 'Not Diabetic'} with confidence {confidence:.2f}%."
        reasoning_text = f"{explanations_text}"

        print(reasoning_text)

        return render_template('index.html', prediction=result_text, explanation=reasoning_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
