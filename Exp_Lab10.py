from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Load the Iris dataset
def load_iris_data():
    iris = datasets.load_iris()
    X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
    y = iris.target  # Labels (Setosa, Versicolour, Virginica)
    return X, y

# Step 2: Train the SVM classifier
def train_svm(X_train, y_train):
    clf = SVC(kernel='linear')  # Use linear kernel for simplicity
    clf.fit(X_train, y_train)
    return clf

# Step 3: Evaluate the SVM model
def evaluate_svm(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    
    correct_predictions = []
    incorrect_predictions = []
    
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct_predictions.append((X_test[i], y_test[i], y_pred[i]))
        else:
            incorrect_predictions.append((X_test[i], y_test[i], y_pred[i]))
    
    return correct_predictions, incorrect_predictions

# Step 4: Function to make predictions for new data points
def predict_svm(clf):
    print("Enter the flower features (sepal length, sepal width, petal length, petal width):")
    try:
        features = list(map(float, input("Enter the features separated by spaces: ").split()))
        prediction = clf.predict([features])
        print(f"Predicted class: {prediction[0]}")
    except ValueError:
        print("Invalid input. Please enter numeric values.")

# Step 5: Main function to execute the program
def main():
    # Load data
    X, y = load_iris_data()
    
    # Step 1: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 2: Train the SVM model
    clf = train_svm(X_train, y_train)
    
    # Step 3: Evaluate the model and print correct and incorrect predictions
    correct_predictions, incorrect_predictions = evaluate_svm(clf, X_test, y_test)
    
    print("\nCorrect Predictions:")
    for sample, actual, predicted in correct_predictions:
        print(f"Sample: {sample}, Actual: {actual}, Predicted: {predicted}")
    
    print("\nIncorrect Predictions:")
    for sample, actual, predicted in incorrect_predictions:
        print(f"Sample: {sample}, Actual: {actual}, Predicted: {predicted}")
    
    # Step 4: Allow the user to input their own data and make a prediction
    while True:
        user_input = input("\nDo you want to predict for a new data point? (yes/no): ").lower()
        if user_input == 'yes':
            predict_svm(clf)
        else:
            print("Exiting the program.")
            break

# Run the main function
if __name__ == "__main__":
    main()

