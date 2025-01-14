import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset (substitute with your own dataset if needed)
data = load_iris()
X = data.data
y = data.target

# Initialize the model (Random Forest as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize StratifiedKFold (ensures class distribution is maintained across folds)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Loop over each fold
for train_index, test_index in kf.split(X, y):
    # Split the data into train and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model on the training set
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Store the metrics for this fold
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Print out the metrics for each fold
print("Metrics per fold:")
for i in range(5):
    print(f"Fold {i+1}:")
    print(f"  Accuracy: {accuracies[i]:.4f}")
    print(f"  Precision: {precisions[i]:.4f}")
    print(f"  Recall: {recalls[i]:.4f}")
    print(f"  F1-score: {f1_scores[i]:.4f}")
    print()

# Compute average of each metric across all folds
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)

# Print out the average metrics
print("Average Metrics:")
print(f"  Average Accuracy: {avg_accuracy:.4f}")
print(f"  Average Precision: {avg_precision:.4f}")
print(f"  Average Recall: {avg_recall:.4f}")
print(f"  Average F1-score: {avg_f1_score:.4f}")

