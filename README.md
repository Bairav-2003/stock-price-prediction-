# stock-price-prediction-

~~~
# Importing libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Loading the dataset
dataset = pd.read_csv('F:/PythonML/Python Datasets/Regression Datasets/tesla.csv')

# Display the first few rows
print(dataset.head())

# Convert 'Date' column to datetime
dataset['Date'] = pd.to_datetime(dataset['Date'])

# Display the shape of the dataset
print(dataset.shape)

# Drop the 'Adj Close' column
dataset.drop('Adj Close', axis=1, inplace=True)

# Display the first few rows after modifications
print(dataset.head())

# Check for missing values
print(dataset.isnull().sum())

# Check if any missing values exist
print(dataset.isna().any())

# Display dataset statistics
print(dataset.describe())

# Print the number of rows in the dataset
print(f"Number of rows in the dataset: {len(dataset)}")

# Plot the 'Open' price
dataset['Open'].plot(figsize=(16, 6))
plt.show()

# Prepare the features and target
X = dataset[['Open', 'High', 'Low', 'Volume']]
y = dataset['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Display the shape of the training data
print(X_train.shape)

# Initialize the Linear Regression model
regressor = LinearRegression()

# Train the model
regressor.fit(X_train, y_train)

# Display the coefficients of the regression model
print("Coefficients:", regressor.coef_)

#


~~~
