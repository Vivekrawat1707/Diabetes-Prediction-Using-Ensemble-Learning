from data_processing.preprocessing import normalize_data
from models import KNNClassifier, SVMClassifier, GradientDescentClassifier
import csv

def load_trained_models(training_data):
    """Load and train all classifiers."""
    models = {
        'KNN': KNNClassifier(k=3),
        'SVM': SVMClassifier(),
        'GD': GradientDescentClassifier()
    }
    
    # Train each model
    for model in models.values():
        model.fit(training_data)
    
    return models

def get_user_input():
    """Get patient data from user input."""
    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]
    
    print("\nPlease enter patient information:")
    print("-" * 40)
    
    values = []
    for feature in features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                values.append(value)
                break
            except ValueError:
                print("Please enter a valid number.")
    
    return values

def predict_diabetes(models, patient_data, min_max_values):
    """Make predictions using all models."""
    # Normalize the patient data
    normalized_patient = []
    for i, value in enumerate(patient_data):
        if min_max_values[i][1] - min_max_values[i][0] == 0:
            normalized_patient.append(0)
        else:
            normalized_patient.append(
                (value - min_max_values[i][0]) / 
                (min_max_values[i][1] - min_max_values[i][0])
            )
    
    # Get predictions from each model
    predictions = {}
    for name, model in models.items():
        pred = model.predict([normalized_patient + [0]])[0]  # Add dummy class
        predictions[name] = pred
    
    return predictions

def main():
    # Load and preprocess training data
    print("Loading training data...")
    dataset = []
    with open('diabetes.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            dataset.append([float(x) for x in row])
    
    # Normalize training data and get normalization parameters
    normalized_data, min_max_values = normalize_data(dataset)
    
    # Train models
    print("Training models...")
    models = load_trained_models(normalized_data)
    
    while True:
        print("\nDiabetes Prediction System")
        print("=" * 40)
        print("1. Enter patient data manually")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == '1':
            # Get patient data
            patient_data = get_user_input()
            
            # Make predictions
            predictions = predict_diabetes(models, patient_data, min_max_values)
            
            # Display results
            print("\nPrediction Results:")
            print("-" * 40)
            for model_name, prediction in predictions.items():
                result = "Diabetic" if prediction == 1 else "Non-diabetic"
                print(f"{model_name:20} : {result}")
            
            # Consensus prediction
            consensus = sum(predictions.values()) > len(predictions) / 2
            print("\nConsensus Prediction:")
            print("-" * 40)
            print("The patient is likely to be:", 
                  "Diabetic" if consensus else "Non-diabetic")
            
        elif choice == '2':
            print("\nThank you for using the Diabetes Prediction System!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()