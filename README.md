# Customer Churn Prediction Pipeline

This project implements a machine learning pipeline for predicting customer churn using a 1D Convolutional Neural Network (CNN) with Keras and NumPy. The pipeline includes data preprocessing (label encoding and normalization), dataset splitting, model building, training, and evaluation.

## Features

- **Label Encoding:** Converts categorical columns to numerical values.
- **Min-Max Normalization:** Scales specified numerical columns to the [0, 1] range.
- **Dataset Splitting:** Splits data into training, validation, and test sets.
- **Model Building:** Constructs a simple 1D CNN for binary classification.
- **Training & Evaluation:** Trains the model and evaluates its performance.

## Requirements

- Python 3.6+
- NumPy
- Keras (with TensorFlow backend)

Notes
The model uses a 1D CNN followed by dense layers for binary classification.
The label column is expected to be at index 19 after preprocessing.
The code modifies the input NumPy array in place during preprocessing.