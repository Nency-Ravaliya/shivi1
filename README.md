# Healthcare Provider Fraud Detection

This project implements a deep learning model using Kolmogorov-Arnold Network (KAN) architecture to detect fraudulent healthcare providers based on Medicare claims data.

## Features

- Custom KAN layer implementation for complex pattern recognition
- Advanced data preprocessing and feature engineering
- Handling of imbalanced data using SMOTE
- Comprehensive model evaluation and visualization
- High accuracy fraud detection (target: 97%)

## Dataset Structure

The system uses four main datasets:
1. Train.csv - Main provider information and fraud labels
2. Train_Beneficiarydata.csv - Patient/beneficiary information
3. Train_Inpatientdata.csv - Inpatient claims data
4. Train_Outpatientdata.csv - Outpatient claims data

## Requirements

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
tensorflow==2.13.0
kaggle==1.5.16
imbalanced-learn==0.11.0
matplotlib==3.7.2
seaborn==0.12.2
```

## Model Architecture

- Custom Kolmogorov-Arnold Network (KAN) layers
- Residual connections
- Batch normalization
- Dropout for regularization
- Dense layers for feature extraction

## Usage

1. Place the dataset files in the `dataset` folder
2. Run the training script:
   ```
   python healthcare_fraud_detection.py
   ```

## Output

The model generates:
- Confusion matrix visualization
- Training history plots
- Detailed classification report
- Model performance metrics

## Model Performance

The model aims to achieve 97% accuracy through:
- Balanced class weights
- SMOTE oversampling
- Advanced feature engineering
- Custom KAN architecture