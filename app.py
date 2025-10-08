from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('stroke_model.pkl')

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        # Numeric and categorical inputs
        age = float(data['age'])
        avg_glucose_level = float(data['avg_glucose_level'])
        bmi = float(data['bmi'])
        hypertension = int(data['hypertension'])
        heart_disease = int(data['heart_disease'])
        gender = int(data['gender'])
        ever_married = int(data['ever_married'])
        Residence_type = int(data['Residence_type'])

        work_type_Govt_job = int(data.get('work_type_Govt_job', 0))
        work_type_Never_worked = int(data.get('work_type_Never_worked', 0))
        work_type_Private = int(data.get('work_type_Private', 0))
        work_type_Self_employed = int(data.get('work_type_Self_employed', 0))

        smoking_status_formerly_smoked = int(data.get('smoking_status_formerly_smoked', 0))
        smoking_status_never_smoked = int(data.get('smoking_status_never_smoked', 0))
        smoking_status_smokes = int(data.get('smoking_status_smokes', 0))

        features = np.array([[age, hypertension, heart_disease, ever_married,
                              avg_glucose_level, bmi, gender, Residence_type,
                              work_type_Govt_job, work_type_Never_worked,
                              work_type_Private, work_type_Self_employed,
                              smoking_status_formerly_smoked, smoking_status_never_smoked,
                              smoking_status_smokes]])

        prob = model.predict_proba(features)[0][1]

        if prob < 0.33:
            result = f"🟢 Low Risk ({prob*100:.1f}%)"
        elif prob < 0.66:
            result = f"🟠 Medium Risk ({prob*100:.1f}%)"
        else:
            result = f"🔴 High Risk ({prob*100:.1f}%)"

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
