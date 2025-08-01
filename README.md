# Customer Churn Prediction Pipeline

This project implements a machine learning pipeline for predicting customer churn using a 1D Convolutional Neural Network (CNN) with Keras and NumPy. The pipeline includes data preprocessing (label encoding and normalization), dataset splitting, model building, training, and evaluation.

## Requirements

- Python 3.6+
- NumPy
- Keras (with TensorFlow backend)

Notes
- The model uses a 1D CNN followed by dense layers for binary classification.
- The label column is expected to be at index 19 after preprocessing.
