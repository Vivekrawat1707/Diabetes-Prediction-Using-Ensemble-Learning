from data_processing import load_data, normalize_data, split_dataset
from models import KNNClassifier, SVMClassifier, GradientDescentClassifier
from utils.metrics import calculate_metrics
import json

def print_dataset_info(dataset):
    """Print basic information about the dataset."""
    features = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Total samples: {len(dataset)}")
    
    for i, feature in enumerate(features):
        values = [row[i] for row in dataset]
        mean = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        print(f"{feature:25} Mean: {mean:.2f} | Min: {min_val:.2f} | Max: {max_val:.2f}")

def evaluate_model(name, model, train_data, test_data):
    """Evaluate a model and print its metrics."""
    print(f"\n{name} Evaluation:")
    print("-" * 50)
    
    # Train the model
    print("Training model...")
    model.fit(train_data)
    
    # Make predictions
    print("Making predictions...")
    y_true = [row[-1] for row in test_data]
    predictions = model.predict(test_data)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, predictions)
    
    # Print results
    print("\nResults:")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1_score']:.2%}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"True Positives:  {cm['true_positives']:4d}    False Positives: {cm['false_positives']:4d}")
    print(f"False Negatives: {cm['false_negatives']:4d}    True Negatives:  {cm['true_negatives']:4d}")
    
    return metrics

def save_metrics(metrics_dict, filename='model_metrics.json'):
    """Save metrics to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\nMetrics saved to {filename}")

def main():
    print("\nDiabetes Prediction using Multiple Classifiers")
    print("=" * 45)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    dataset = load_data('diabetes.csv')
    print_dataset_info(dataset)
    
    normalized_data, _ = normalize_data(dataset)
    train_data, test_data = split_dataset(normalized_data)
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Initialize classifiers
    classifiers = [
        ("K-Nearest Neighbors (k=3)", KNNClassifier(k=3)),
        ("Support Vector Machine", SVMClassifier(learning_rate=0.01)),
        ("Gradient Descent", GradientDescentClassifier(learning_rate=0.01))
    ]
    
    # Evaluate each classifier and store metrics
    all_metrics = {}
    for name, classifier in classifiers:
        metrics = evaluate_model(name, classifier, train_data, test_data)
        all_metrics[name] = metrics
    
    # Save metrics to file
    save_metrics(all_metrics)
    
    # Print overall comparison
    print("\nModel Comparison Summary:")
    print("-" * 50)
    print(f"{'Model':30} {'Accuracy':10} {'F1 Score':10}")
    print("-" * 50)
    for name, metrics in all_metrics.items():
        print(f"{name:30} {metrics['accuracy']:.2%}    {metrics['f1_score']:.2%}")

if __name__ == "__main__":
    main()