# CodSoft Internship – Machine Learning (NOV Batch B63)

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Internship-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 👨‍💻 Intern Information
- **Name:** Sarthak Srivastava  
- **Position:** Machine Learning Intern  
- **Duration:** 10 Nov 2025 – 10 Dec 2025 (4 Weeks)  
- **Batch:** NOV Batch B63

---

## 📋 Table of Contents
- [Overview](#overview)
- [Projects](#projects)
  - [1. Credit Card Fraud Detection](#1-credit-card-fraud-detection)
  - [2. Movie Genre Classification](#2-movie-genre-classification)
  - [3. SMS Spam Detection](#3-sms-spam-detection)
- [Technologies Used](#technologies-used)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Internship Compliance](#internship-compliance)
- [Learning Outcomes](#learning-outcomes)
- [Contact](#contact)

---

## 🎯 Overview

This repository contains all the machine learning projects completed during my internship at **CodSoft**. The internship focused on building practical ML models for real-world applications including fraud detection, text classification, and natural language processing.

All three mandatory projects have been successfully completed, demonstrating proficiency in:
- Data preprocessing and exploratory data analysis
- Feature engineering and selection
- Model training and evaluation
- Handling imbalanced datasets
- Text processing and NLP techniques

---

## 🚀 Projects

### 1. Credit Card Fraud Detection
**Objective:** Build a machine learning model to identify fraudulent credit card transactions.

**Key Features:**
- Handles highly imbalanced dataset (fraud vs. legitimate transactions)
- Implements multiple classification algorithms (Logistic Regression, Decision Trees, Random Forest)
- Uses SMOTE for handling class imbalance
- Comprehensive evaluation metrics (Precision, Recall, F1-Score, ROC-AUC)

**Technologies:**
- Python, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn for visualization
- Imbalanced-learn for SMOTE

**Dataset:** Credit card transaction data with features anonymized using PCA

📥 **[Download Dataset from Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)**

📂 **[View Project](./credit%20card%20detection/)**

---

### 2. Movie Genre Classification
**Objective:** Predict movie genres based on plot descriptions using NLP and machine learning.

**Key Features:**
- Text preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization for feature extraction
- Multi-class classification
- Model comparison and performance analysis

**Technologies:**
- Python, Pandas, NumPy, Scikit-learn
- NLTK/SpaCy for text processing
- TfidfVectorizer for feature extraction

**Dataset:** Movie plot descriptions with genre labels

� **[Download Dataset from Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)**

�📂 **[View Project](./MOVIE%20GENRE%20CLASSIFICATION/)**

---

### 3. SMS Spam Detection
**Objective:** Classify SMS messages as spam or legitimate (ham) using machine learning.

**Key Features:**
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Multiple classifier comparison (Naive Bayes, Logistic Regression, SVM)
- Model performance evaluation and optimization

**Technologies:**
- Python, Pandas, NumPy, Scikit-learn
- Natural Language Processing techniques
- TF-IDF vectorization

**Dataset:** SMS message dataset with spam/ham labels

� **[Download Dataset from Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)**

�📂 **[View Project](./SMS%20SPAM%20DETECTOR/)**

---

## 🛠️ Technologies Used

### Programming Languages
- Python 3.8+

### Libraries & Frameworks
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Imbalanced-learn
- **Natural Language Processing:** NLTK, SpaCy
- **Visualization:** Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebook

### Algorithms Implemented
- Logistic Regression
- Decision Trees
- Random Forest
- Naive Bayes
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)

---

## 📁 Repository Structure

```
codsoft/
│
├── credit card detection/
│   ├── README.md
│   ├── credit_card_fraud_detection_ENHANCED.ipynb
│   ├── fraudTrain.csv
│   └── fraudTest.csv
│
├── MOVIE GENRE CLASSIFICATION/
│   ├── README.md
│   ├── movie_genre_classifier.ipynb
│   ├── requirements.txt
│   ├── predictions.txt
│   └── Genre Classification Dataset/
│
├── SMS SPAM DETECTOR/
│   ├── README.md
│   ├── sms_spam_detection.ipynb
│   ├── requirements.txt
│   └── spam.csv
│
└── README.md (this file)
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### General Setup Steps

> **⚠️ Important Note About Dataset Files:**  
> The dataset files (CSV files) are very large (500+ MB total) and are **NOT included in this GitHub repository** to avoid upload issues. You must download them separately from the Kaggle links provided in each project section above.

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd codsoft
   ```

2. **Download the datasets**
   - Credit Card: [Download from Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
   - Movie Genre: [Download from Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
   - SMS Spam: [Download from Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
   
   Place the downloaded datasets in their respective project folders.

3. **Navigate to specific project**
   ```bash
   cd "project-folder-name"
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   *Note: If requirements.txt is not present, install common dependencies:*
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter nltk imbalanced-learn
   ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

6. **Open and run the .ipynb file**

### Project-Specific Instructions
Each project folder contains its own README.md with detailed setup and execution instructions.

---

## ✅ Internship Compliance

This internship submission adheres to all CodSoft requirements:

- ✅ **Minimum Projects:** 3 out of 3 completed
- ✅ **Original Code:** All implementations are original and not copied
- ✅ **Separate Folders:** Each task maintained in dedicated directories
- ✅ **Video Demonstrations:** Shared on LinkedIn tagging @CodSoft
- ✅ **GitHub Repository:** Submitted via official task form
- ✅ **Documentation:** Comprehensive README files for each project

### 🔗 Submission Assets
- **GitHub Repository:** Submitted in official form
- **LinkedIn Posts:** 
  - Offer Letter announcement
  - Project 1 demo video (Credit Card Fraud Detection)
  - Project 2 demo video (Movie Genre Classification)
  - Project 3 demo video (SMS Spam Detection)

---

## 📚 Learning Outcomes

Through this internship, I gained hands-on experience in:

1. **Data Science Workflow**
   - Data collection, cleaning, and preprocessing
   - Exploratory Data Analysis (EDA)
   - Feature engineering and selection

2. **Machine Learning**
   - Supervised learning algorithms
   - Model training, validation, and testing
   - Hyperparameter tuning
   - Performance evaluation metrics

3. **Specialized Techniques**
   - Handling imbalanced datasets
   - Natural Language Processing (NLP)
   - Text classification and sentiment analysis
   - TF-IDF vectorization

4. **Best Practices**
   - Code documentation and version control
   - Project organization and structure
   - Model evaluation and comparison
   - Reproducible research practices

---


## 📄 License

This repository is maintained for internship evaluation and learning purposes.

---

## 🙏 Acknowledgments

Special thanks to **CodSoft** for providing this learning opportunity and guidance throughout the internship program.

---

<div align="center">
  <strong>⭐ If you found this repository helpful, please consider giving it a star! ⭐</strong>
</div>
