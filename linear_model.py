import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Define paths to the dataset files
file_red = "winequality-red.csv"
file_white = "winequality-white.csv"

def load_dataset(file_name):
    file_path = os.path.join("/opt/ml/input/data/train", file_name)
    dataset = pd.read_csv(file_path, delimiter=';', engine='python')
    return dataset

# Load datasets
dataset_red = load_dataset(file_red)
dataset_white = load_dataset(file_white)

# Combine datasets
dataset = pd.concat([dataset_red, dataset_white], axis=0)

# Check for and handle missing values or invalid data
if dataset.isnull().sum().any():
    print("Missing values detected. Dropping rows with missing values.")
    dataset = dataset.dropna()

# Split features and target
X = dataset.drop(columns=['quality'])
y = dataset['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save the model
model_path = '/opt/ml/model/linear_regression_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
