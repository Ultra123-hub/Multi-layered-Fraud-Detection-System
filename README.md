# Multi-layered-Fraud-Detection-System

# Fraud Detection System — End-to-End Machine Learning Pipeline

## 📌 Project Overview
This project implements an end-to-end **fraud detection system** using a combination of **supervised** and **unsupervised** machine learning models. The goal is to accurately classify users as **fraudulent** or **non-fraudulent** by learning patterns from user activity and trading behavior, while addressing real-world challenges such as **class imbalance**, **feature skewness**, and **deployment readiness**. 

The pipeline covers:
- Feature engineering from raw transactional data  
- Data preprocessing and imbalance handling  
- Training and evaluation of multiple models  
- Model ensembling for improved performance  
- Preparation for deployment using **FastAPI**, **Docker**, and cloud platforms  

---

## 🧠 Models Implemented

### Supervised Models
- Logistic Regression  
- Random Forest  
- XGBoost  
- Multi-Layer Perceptron (MLP)

### Unsupervised Models
- Autoencoder  
- Variational Autoencoder (VAE)

### Ensemble Strategy
- **Stacking ensemble** combining predictions from supervised models to improve robustness and generalization.

---

## 🧪 Evaluation Metrics
Each model is evaluated using metrics suitable for **imbalanced classification problems**, including:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- PR-AUC  

Special emphasis is placed on **Recall** and **PR-AUC**, as missing fraudulent cases is more costly than false positives in real-world fraud systems.

---

## 📊 Final Model Performance Summary

### Supervised Models (2 d.p.)

| Model                 | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|----------------------|----------|---------|--------|-----------|--------|----------|
| Logistic Regression  | 0.90     | 0.98    | 0.91   | 0.74      | 1.00   | 0.85     |
| Random Forest        | 0.98     | 1.00    | 1.00   | 0.93      | 1.00   | 0.97     |
| XGBoost              | 0.99     | 1.00    | 1.00   | 0.96      | 1.00   | 0.98     |
| MLP                  | 0.95     | 0.99    | 0.99   | 0.85      | 1.00   | 0.92     |

### Unsupervised Models

| Model        | Accuracy | ROC-AUC | PR-AUC |
|-------------|----------|---------|--------|
| Autoencoder | 0.70     | 0.77    | 0.50   |
| VAE         | 0.70     | 0.62    | 0.39   |

### Ensemble Model

- **Accuracy:** 0.98  
- **Precision:** 1.00  
- **Recall:** 0.94  
- **F1-Score:** 0.97  
- **ROC-AUC:** 1.00  

The ensemble achieves the best balance between **precision and recall**, making it the most reliable candidate for deployment.

---

## ⚙️ Data Processing Workflow
1. Feature aggregation from user activity and trades  
2. Handling missing values  
3. Log-transformation of highly skewed numerical features  
4. Train-test split  
5. Feature scaling (fit on training set only)  
6. Class imbalance handling (class weights / resampling)  
7. Model training and evaluation  
8. Model ensembling  
9. Saving scalers and models for inference  

---

## 🚀 Deployment (Coming Next)
The trained models and preprocessing pipeline are prepared for production deployment.  
The **next article** in this series will cover:
- Serving the model with **FastAPI**  
- Containerization using **Docker**  
- Deployment to a **cloud platform** (AWS / Azure / Heroku)  

---

## 📁 Project Structure
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── feature_engineering.ipynb
│   ├── supervised_models.ipynb
│   ├── unsupervised_models.ipynb
│   └── ensemble_model.ipynb
├── models/
│   ├── saved_models/
│   └── scalers/
├── api/
│   └── main.py
├── Dockerfile
├── requirements.txt
└── README.md


