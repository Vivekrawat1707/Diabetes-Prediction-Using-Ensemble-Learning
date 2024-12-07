import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()

def plot_feature_distributions(df):
    """Plot feature distributions by outcome."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.ravel()
    
    for idx, column in enumerate(df.columns[:-1]):
        sns.boxplot(x='Outcome', y=column, data=df, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {column} by Outcome')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

def plot_model_comparison(model_accuracies):
    """Plot model accuracy comparison."""
    plt.figure(figsize=(10, 6))
    plt.bar(model_accuracies.keys(), model_accuracies.values())
    plt.title('Model Comparison - Accuracy Scores')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for i, v in enumerate(model_accuracies.values()):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
    
    plt.savefig('model_comparison.png')
    plt.close()