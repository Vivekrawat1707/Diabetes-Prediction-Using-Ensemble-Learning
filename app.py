from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from data_processing.preprocessing import normalize_data
from models import KNNClassifier, SVMClassifier, GradientDescentClassifier
import csv

app = Flask(__name__)
CORS(app)

# Load dataset and models
dataset = []
with open('diabetes.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        dataset.append([float(x) for x in row])

normalized_data, min_max_values = normalize_data(dataset)
models = {
    'KNN': KNNClassifier(k=3),
    'SVM': SVMClassifier(),
    'GD': GradientDescentClassifier()
}
for model in models.values():
    model.fit(normalized_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        patient_data = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodPressure']),
            float(data['skinThickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetesPedigreeFunction']),
            float(data['age'])
        ]
        normalized_patient = [
            (value - min_max_values[i][0]) / (min_max_values[i][1] - min_max_values[i][0])
            if min_max_values[i][1] - min_max_values[i][0] != 0 else 0
            for i, value in enumerate(patient_data)
        ]
        predictions = {
            name: "Diabetic" if model.predict([normalized_patient + [0]])[0] == 1 else "Non-diabetic"
            for name, model in models.items()
        }
        consensus = "Diabetic" if sum(1 for pred in predictions.values() if pred == "Diabetic") > len(predictions) / 2 else "Non-diabetic"
        return jsonify({'predictions': predictions, 'consensus': consensus})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
