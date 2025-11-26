# Credit Card Fraud Detection System

A machine learning-based fraud detection system that classifies credit card transactions as fraudulent or legitimate using Random Forest, Decision Tree, and Logistic Regression algorithms.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Files in This Project](#files-in-this-project)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Available Transaction Categories](#available-transaction-categories)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project implements a credit card fraud detection system using machine learning. The system analyzes transaction features (category, amount, gender) to predict whether a transaction is fraudulent or legitimate.

**Best Model:** Random Forest Classifier
- **Accuracy:** 99.7%
- **Fraud Detection Rate (Recall):** ~59%
- **Precision:** ~62%
- **ROC-AUC Score:** ~0.95

---

## ✨ Features

- ✅ **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Easy Predictions**: Simple functions to check single or multiple transactions
- ✅ **Rich Visualizations**: 
  - Confusion Matrix Heatmap
  - ROC Curve with AUC score
  - Feature Importance plots
- ✅ **Batch Processing**: Check multiple transactions at once
- ✅ **Hindi Documentation**: Complete instructions in Hindi

---

## 📁 Files in This Project

### Main Files

| File | Description | Size |
|------|-------------|------|
| `credit_card_fraud_detection_ENHANCED.ipynb` | **⭐ USE THIS!** Enhanced notebook with all features | 53 KB |
| `credit_card_fraud_detection.ipynb` | Original notebook (basic version) | 39 KB |

### Dataset Files

**Source:** [Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

> **⚠️ Note:** The dataset files are NOT included in the GitHub repository due to their large size (~500 MB). Please download them from the Kaggle link above.

| File | Description | Size |
|------|-------------|------|
| `fraudTrain.csv` | Training dataset (1.3M transactions) | 335 MB |
| `fraudTest.csv` | Test dataset (555K transactions) | 143 MB |

### Supporting Files

| File | Description |
|------|-------------|
| `use_fraud_detector.py` | Standalone Python script (optional) |
| `README_HINDI.md` | Complete documentation in Hindi |
| `README.md` | This file |

### Generated Files (After Running Notebook)

These files are created automatically when you run the notebook:

- `fraud_model.pkl` - Trained Random Forest model
- `scaler.pkl` - StandardScaler for amount normalization
- `label_encoder_category.pkl` - Encoder for transaction categories
- `label_encoder_gender.pkl` - Encoder for gender

---

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook

### Install Required Packages

> **⚠️ IMPORTANT: Dataset Download Required**
> The dataset files (`fraudTrain.csv` and `fraudTest.csv`) are NOT included in this repository because they are too large (~500 MB).
> 
> **You must download them from Kaggle:** [Click Here to Download](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
> 
> After downloading, extract the files into this folder before running the notebook.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Step 1: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 2: Open the Enhanced Notebook

Navigate to and open: `credit_card_fraud_detection_ENHANCED.ipynb`

### Step 3: Run All Cells

- Click **Cell → Run All**
- Or press **Shift + Enter** on each cell

### Step 4: Start Predicting!

Scroll to the bottom of the notebook and use the prediction functions.

---

## 📖 Usage Guide

### Single Transaction Prediction

```python
# Check if a transaction is fraudulent
predict_fraud(
    category='grocery_pos',  # Transaction category
    amount=100.00,           # Transaction amount
    gender='M'               # Customer gender ('M' or 'F')
)
```

**Output:**
```
==================================================
Transaction Details:
  Category: grocery_pos
  Amount: $100.00
  Gender: M
--------------------------------------------------
✓  Transaction is LEGITIMATE
  Legitimate Probability: 98.50%
==================================================
```

### Batch Prediction

```python
# Check multiple transactions at once
transactions = [
    {'category': 'gas_transport', 'amount': 45.50, 'gender': 'M'},
    {'category': 'grocery_pos', 'amount': 120.75, 'gender': 'F'},
    {'category': 'misc_net', 'amount': 500.00, 'gender': 'M'}
]

results_df = check_multiple_transactions(transactions)
```

### Using Saved Models (Optional)

```python
import pickle

# Load the saved model
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

---

## 📊 Model Performance

### Random Forest (Best Model)

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.7% | Overall correctness |
| **Precision** | 62% | When flagged as fraud, 62% are actually fraud |
| **Recall** | 59% | Detects 59% of all frauds |
| **F1-Score** | 0.61 | Harmonic mean of precision and recall |
| **ROC-AUC** | 0.95 | Excellent discrimination ability |

### Confusion Matrix Results

- **True Negatives:** 552,795 (Legitimate correctly identified)
- **False Positives:** 779 (False alarms - 0.14%)
- **False Negatives:** 872 (Missed frauds - 41%)
- **True Positives:** 1,273 (Frauds correctly detected - 59%)

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 99.7% | 0.62 | 0.59 | 0.61 |
| Decision Tree | 99.7% | 0.61 | 0.59 | 0.60 |
| Logistic Regression | 99.6% | 0.00 | 0.00 | 0.00 |

**Note:** Logistic Regression fails to detect frauds due to class imbalance.

---

## 🏷️ Available Transaction Categories

The model recognizes the following transaction categories:

| Category | Description |
|----------|-------------|
| `gas_transport` | Gas station/transportation |
| `grocery_pos` | Grocery store purchases |
| `misc_net` | Online miscellaneous |
| `misc_pos` | Miscellaneous point of sale |
| `entertainment` | Entertainment expenses |
| `food_dining` | Food and dining |
| `personal_care` | Personal care products |
| `health_fitness` | Health and fitness |
| `travel` | Travel expenses |
| `kids_pets` | Kids and pets |
| `shopping_net` | Online shopping |
| `shopping_pos` | In-store shopping |
| `home` | Home-related expenses |

---

## 💡 Examples

### Example 1: Gas Station Purchase (Legitimate)

```python
predict_fraud('gas_transport', 45.50, 'M')
```

**Expected Result:** ✓ LEGITIMATE

### Example 2: Large Online Purchase (Potentially Fraudulent)

```python
predict_fraud('misc_net', 500.00, 'M')
```

**Expected Result:** ⚠️ FRAUD DETECTED (depending on model)

### Example 3: Grocery Shopping (Legitimate)

```python
predict_fraud('grocery_pos', 120.75, 'F')
```

**Expected Result:** ✓ LEGITIMATE

### Example 4: Batch Check

```python
sample_transactions = [
    {'category': 'gas_transport', 'amount': 45.50, 'gender': 'M'},
    {'category': 'grocery_pos', 'amount': 120.75, 'gender': 'F'},
    {'category': 'misc_net', 'amount': 500.00, 'gender': 'M'},
    {'category': 'entertainment', 'amount': 250.00, 'gender': 'F'}
]

results = check_multiple_transactions(sample_transactions)
print(results)
```

---

## 🔍 Troubleshooting

### Problem: "NameError: name 'le_category' is not defined"

**Solution:** Run the "Save Models" cell first to create the label encoders.

### Problem: "FileNotFoundError: fraud_model.pkl"

**Solution:** Run all cells in the notebook from the beginning to train and save the model.

### Problem: "ValueError: y should be a 1d array"

**Solution:** Check your input format. Use:
```python
predict_fraud('category_name', amount, 'M' or 'F')
```

### Problem: "KeyError: 'category_name'"

**Solution:** Make sure you're using a valid category from the [Available Categories](#available-transaction-categories) list.

### Problem: Module not found errors

**Solution:** Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🎓 Future Improvements

### To Improve Detection Rate (Currently 59%)

1. **Add More Features:**
   - Transaction time (hour of day, day of week)
   - Location distance (customer to merchant)
   - Transaction frequency
   - Historical spending patterns

2. **Handle Class Imbalance:**
   - Use SMOTE (Synthetic Minority Over-sampling Technique)
   - Adjust class weights in the model
   - Try ensemble methods

3. **Try Advanced Models:**
   - XGBoost
   - LightGBM
   - Neural Networks (Deep Learning)
   - Isolation Forest (Anomaly Detection)

4. **Hyperparameter Tuning:**
   - GridSearchCV
   - RandomizedSearchCV
   - Bayesian Optimization

5. **Feature Engineering:**
   - Create derived features (e.g., amount deviation from average)
   - Time-based features
   - Merchant risk scores

---

## 📚 Documentation

- **English:** This README
- **Hindi:** `README_HINDI.md`

---

## 🤝 Contributing

This is a learning project. Feel free to:
- Experiment with different models
- Add new features
- Improve the detection rate
- Create visualizations

---

## 📄 License

This project is for educational purposes.

---

## 👨‍💻 Author

Created as part of CodSoft internship project.

---

## 📞 Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the Hindi documentation in `README_HINDI.md`
3. Ensure all cells are run in order

---

## 🎉 Quick Reference

### Key Functions

```python
# Single prediction
predict_fraud(category, amount, gender)

# Batch prediction
check_multiple_transactions(transactions_list)
```

### Key Files

- **Main Notebook:** `credit_card_fraud_detection_ENHANCED.ipynb`
- **Model File:** `fraud_model.pkl` (generated after running)
- **Hindi Docs:** `README_HINDI.md`

### Important Notes

- ⚠️ Model misses ~41% of frauds (recall = 59%)
- ✅ Very low false alarm rate (0.14%)
- 📊 Best for high-value transaction monitoring
- 🔄 Can be improved with more features

---

**Happy Fraud Detection! 🎉**

For detailed Hindi instructions, see: `README_HINDI.md`
