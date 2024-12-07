import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from models import KNNClassifier, SVMClassifier, GradientDescentClassifier
from visualization import plot_correlation_matrix, plot_feature_distributions, plot_model_comparison

def main():
    print("\nDiabetes Prediction Analysis")
    print("=" * 30)
    
    # Load and explore data
    print("\n1. Loading and exploring data...")
    df = pd.read_csv('diabetes.csv')
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Create visualizations
    print("\n2. Generating visualizations...")
    plot_correlation_matrix(df)
    plot_feature_distributions(df)
    print("   - Correlation matrix saved as 'correlation_matrix.png'")
    print("   - Feature distributions saved as 'feature_distributions.png'")
    
    # Prepare data
    print("\n3. Preparing data for modeling...")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate models
    print("\n4. Training and evaluating models...")
    models = {
        'KNN (k=3)': KNNClassifier(k=3),
        'SVM': SVMClassifier(),
        'Gradient Descent': GradientDescentClassifier()
    }
    
    accuracies = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        
        print(f"Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Plot model comparison
    plot_model_comparison(accuracies)
    print("\n5. Model comparison plot saved as 'model_comparison.png'")

if __name__ == "__main__":
    main()