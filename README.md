# ❤️ Heart Disease Risk Prediction System

An end-to-end **Machine Learning classification system** that predicts the risk of heart disease using clinical and patient-related health parameters.

The project compares multiple classification algorithms, evaluates their performance using cross-validation and ROC-AUC metrics, and provides a real-time prediction interface using Streamlit.

## 🚀 Live Demo

👉 **[Open Heart Disease Risk Prediction App](YOUR_STREAMLIT_APP_URL)**

## ✨ Features

- 🫀 Heart disease risk prediction using clinical parameters
- 📊 Exploratory data analysis and preprocessing
- 🧹 Data cleaning and outlier handling
- ⚖️ Feature scaling and preprocessing
- 🤖 Comparison of multiple ML classification models
- 🔄 5-Fold Cross-Validation
- 📈 ROC-AUC based model evaluation
- 🎯 Real-time prediction through Streamlit
- 📊 Risk probability assessment
- 🖥️ Interactive and user-friendly interface

## 🧠 Machine Learning Models

The following classification algorithms were trained and evaluated:

- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- Logistic Regression

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 🏆 **Random Forest** | **0.9239** | **0.9231** | **0.9412** | **0.9320** | **0.9729** |
| SVM | 0.9185 | 0.9223 | 0.9314 | 0.9268 | 0.9619 |
| Gradient Boosting | 0.8967 | 0.8879 | 0.9314 | 0.9091 | 0.9579 |
| Logistic Regression | 0.8859 | 0.8857 | 0.9118 | 0.8986 | 0.9436 |

### 🏆 Best Performing Model

**Random Forest** achieved the best overall performance:

- **Accuracy:** 92.39%
- **Precision:** 92.31%
- **Recall:** 94.12%
- **F1 Score:** 93.20%
- **ROC-AUC:** 97.29%

The high ROC-AUC indicates strong discrimination between patients with and without heart disease in the evaluated dataset.

## 🔬 Data Preprocessing

The preprocessing pipeline includes:

- Handling missing or invalid values
- Outlier detection and removal
- Feature scaling
- Preparation of clinical and demographic features
- Train-test data splitting

## 🩺 Prediction System

The Streamlit application allows users to enter patient health parameters such as:

- Age
- Sex
- Blood Pressure
- Cholesterol
- ECG-related measurements
- Maximum Heart Rate
- Other clinical parameters

The application then provides a **predicted heart disease risk and probability assessment**.

> **Note:** This application is intended for educational and demonstration purposes only and should not be used as a substitute for professional medical diagnosis or advice.

## 🔄 Machine Learning Workflow

```text
Patient Dataset
       ↓
Data Cleaning
       ↓
Outlier Handling
       ↓
Feature Scaling
       ↓
Train-Test Split
       ↓
5-Fold Cross-Validation
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Random Forest Selection
       ↓
Streamlit Prediction App
       ↓
Heart Disease Risk Probability