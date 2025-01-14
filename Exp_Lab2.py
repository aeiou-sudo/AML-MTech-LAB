import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the Iris dataset as a pandas DataFrame
iris = load_iris(as_frame=True)
df = iris['frame']  # Get the DataFrame

# Use the correct column names
X = df.drop('target', axis=1)  # 'target' represents the species
y = df['target']  # Species is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split successfully!")
print(X)

model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Model trained successfully!")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy * 100)




while True:
    user_input_1 = float(input("Enter the value for sepal_length: "))
    user_input_2 = float(input("Enter the value for sepal_width: "))
    user_input_3 = float(input("Enter the value for petal_length: "))
    user_input_4 = float(input("Enter the value for petal_width: "))

    # Define feature columns
    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # Create a DataFrame for the user input
    user_input = pd.DataFrame([[user_input_1, user_input_2, user_input_3, user_input_4]], columns=feature_columns)

    # Predict the class using the trained model
    pred = model.predict(user_input)

    print(f"The predicted class is: {pred[0]}")

    # Ask the user if they want to make another prediction or exit
    choice = input("Do you want to make another prediction? (yes/no): ").strip().lower()
    if choice != 'yes':
        print("Exiting the prediction loop.")
        break


