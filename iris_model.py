# iris_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data_from_csv(csv_path="C:/Users/aryaa/OneDrive/Desktop/Iris-Classification/IRIS.csv"):
    import pandas as pd

    df = pd.read_csv(csv_path)
    
    # Map species names to integers: 0, 1, 2
    label_map = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    df['species'] = df['species'].map(label_map)

    X = df.drop('species', axis=1).values
    y = df['species'].values
    feature_names = df.columns[:-1]
    target_names = ['setosa', 'versicolor', 'virginica']

    return X, y, feature_names, target_names

def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.1, random_state=42)

def train_svm(X_train, y_train):
    model = SVC(gamma='auto')
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_names, title="SVM"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    print(f"\n--- {title} Model Evaluation ---")
    print("Accuracy:", acc)
    print("Classification Report:\n", report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_{title.lower()}.png")
    plt.close()

def feature_importance_rf(rf_model, feature_names):
    importances = rf_model.feature_importances_
    importance_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\n--- Feature Importances (Random Forest) ---")
    print(importance_df)

def main():
    X, y, feature_names, target_names = load_data_from_csv()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # SVM model
    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, target_names, title="SVM")

    # Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, target_names, title="RandomForest")
    feature_importance_rf(rf_model, feature_names)

if __name__ == "__main__":
    main()
