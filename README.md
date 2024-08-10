# Sales Prediction Using Neural Networks

## Overview

This project involves predicting sales based on various features using a neural network. The dataset used includes information about items and outlets, with the goal of forecasting item sales. The approach includes data preprocessing, feature engineering, normalization, and training a neural network model.

## Project Structure

The project involves the following steps:

1. **Data Loading and Inspection**: Load the dataset and inspect its structure and missing values.
2. **Data Preprocessing**: Clean and preprocess the data by handling missing values and mapping categorical variables.
3. **Feature Normalization**: Normalize the feature columns using Min-Max normalization.
4. **Model Training**: Train a neural network model using TensorFlow and Keras.
5. **Model Evaluation**: Evaluate the model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared score.
6. **Visualization**: Plot the training and validation loss to monitor the training process.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- TensorFlow

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib tensorflow
```
# Instructions
**1. Data Loading and Inspection**

Load the dataset and inspect its structure:

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\fnn09\\Downloads\\train_v9rqX0R.xls")
```
# Display the first few rows of the dataset
df.head()

# Check unique values and missing values
df["Item_Identifier"].value_counts()
df.shape
df["Item_Fat_Content"].unique()
df['Item_Type'].unique()
df['Outlet_Establishment_Year'].unique()
df['Outlet_Size'].isnull().sum()
df['Item_Weight'].isnull().sum()

**2. Data Preprocessing**

Drop redundant features and handle missing values:

```python

# Dropping redundant features
df.drop(['Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1, inplace=True)
df.drop(['Item_Identifier'], axis=1, inplace=True)

# Fill missing values
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].interpolate())
```
**3. Feature Mapping**

Convert categorical columns to numerical values:

```python

df['Item_Fat_Content'] = df['Item_Fat_Content'].map({'Low Fat': 0, 'Regular': 1, 'LF': 2, 'reg': 3, 'low fat': 4})
df['Item_Type'] = df['Item_Type'].map({'Fruits and Vegetables': 0, 'Snack Foods': 1, 'Household': 2, 'Frozen Foods': 3, 'Dairy': 4, 'Canned': 5, 'Baking Goods': 6, 'Health and Hygiene': 7, 'Soft Drinks': 8, 'Meat': 9, 'Breads': 10, 'Hard Drinks': 11, 'Others': 12, 'Starchy Foods': 13, 'Breakfast': 14, 'Seafood': 15})
df['Outlet_Size'] = df['Outlet_Size'].map({'Medium': 0, 'Small': 1, 'High': 2})
df['Outlet_Type'] = df['Outlet_Type'].map({'Supermarket Type1': 0, 'Grocery Store': 1, 'Supermarket Type3': 2, 'Supermarket Type2': 3})
df['Outlet_Location_Type'] = df['Outlet_Location_Type'].map({'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2})
```
**4. Feature Normalization**

Normalize feature columns:

```python

# Normalize feature columns
col = ["Item_Fat_Content", "Item_Type", "Outlet_Size", "Outlet_Type", "Outlet_Location_Type", "Item_Weight", "Item_Outlet_Sales", "Item_MRP"]
for i in col:
    df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())
```
**5. Model Training**

Train a neural network model using TensorFlow and Keras:

```python

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# Split the data into training and testing sets
X = df.drop(['Item_Outlet_Sales'], axis=1)
Y = df['Item_Outlet_Sales']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Build the neural network model
model = Sequential([
    InputLayer(input_shape=(8,)),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    Dense(5, activation='relu'),
    Dense(2, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
model_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)
```
**6. Model Evaluation**

Evaluate the model performance and visualize the loss:

```python

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model predictions
model_predictions = model.predict(x_test)
print(mean_absolute_error(y_test, model_predictions))
print(mean_squared_error(y_test, model_predictions))
print("X test R2 score: ", r2_score(y_test, model_predictions))

model_predictions_train = model.predict(x_train)
print("X train R2 score: ", r2_score(y_train, model_predictions_train))

# Plot training and validation loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```
**License**

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements

    TensorFlow and Keras for building and training the neural network.
    Scikit-learn for metrics and data splitting.
    Matplotlib for visualization.
