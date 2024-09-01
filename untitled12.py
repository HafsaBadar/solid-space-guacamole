

import pickle # Import the pickle module

with open('StudentsPerformance.pkl', 'wb') as f:
    pickle.dump((model, LinearRegression), f) # Use the imported module

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import numpy as np

# Load the trained Linear Regression model from the pickle file
with open('StudentsPerformance,pkl', 'rb') as f:
    model, LinearRegression = pickle.load(f)

# Load the trained Linear Regression model from the pickle file
with open('StudentsPerformance.pkl', 'rb') as f: # Fix the filename
    model, LinearRegression = pickle.load(f)

# Load the trained model from the pickle file
with open('StudentsPerformance.pkl', 'rb') as f:
    model, _ = pickle.load(f)  # Assuming you don't need the second returned value

# Get the expected feature names from the model
expected_feature_names = model.feature_names_in_

# Ensure 'df' has the same feature names
df = df[expected_feature_names]

# Use the loaded model for prediction
prediction = model.predict(df)

# Check if the model is a classification model
if hasattr(model, 'predict_proba'):
    prediction_proba = model.predict_proba(df)
    # ... rest of your code for handling prediction probabilities
else:
    print("Model does not support predict_proba. It might be a regression model.")
    # ... code to handle the case where you have a regression model

# Load the trained Linear Regression model from the pickle file
with open('StudentsPerformance.pkl', 'rb') as f:
    model, LinearRegression = pickle.load(f)

# Get the expected feature names from the model
expected_feature_names = model.feature_names_in_  # Assuming your model has this attribute

# Ensure 'df' has the same feature names
df = df[expected_feature_names]  # Reorder columns to match expected names

# Use the loaded model for prediction
prediction = model.predict(df)