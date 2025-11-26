# 📱 SMS Spam Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An intelligent AI-powered SMS spam detection system that classifies text messages as **spam** or **legitimate (ham)** using machine learning algorithms. This project implements and compares three different classifiers to achieve >95% accuracy.

---

## 🎯 Project Overview

This project builds a comprehensive spam detection system using:
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Machine Learning Models**:
  - Naive Bayes Classifier
  - Logistic Regression
  - Support Vector Machine (SVM)
- **Dataset**: SMS Spam Collection (5,574 messages)
- **Performance**: >95% accuracy across all models

---

## ✨ Features

✅ **Complete Data Pipeline**: From raw data to trained models  
✅ **Exploratory Data Analysis**: Visualizations and statistical insights  
✅ **Text Preprocessing**: Advanced NLP techniques (lemmatization, stopword removal)  
✅ **Three ML Classifiers**: Naive Bayes, Logistic Regression, SVM  
✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score  
✅ **Visual Analytics**: Confusion matrices, word clouds, comparison charts  
✅ **Interactive Testing**: Test custom messages with all three models  
✅ **Consensus Prediction**: Voting system across models  

---

## 📁 Project Structure

```
SMS SPAM DETECTOR/
│
├── sms_spam_detection.ipynb      # Main Jupyter Notebook (Complete Implementation)
├── requirements.txt              # Python dependencies
├── INSTALL.md                    # Installation troubleshooting guide
└── README.md                     # This file
```

> **⚠️ Note:** The `spam.csv` dataset file is NOT included in the GitHub repository due to size constraints. Please download it from Kaggle.

**Dataset Source:** [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies

> **⚠️ IMPORTANT: Dataset Download Required**
> The dataset file (`spam.csv`) is NOT included in this repository.
> 
> **You must download it from Kaggle:** [Click Here to Download](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
> 
> After downloading, place the `spam.csv` file in this folder.

Open a terminal/command prompt and run:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk jupyter
```

> **Note**: If you encounter issues installing `wordcloud`, see [INSTALL.md](INSTALL.md) for troubleshooting.

### 3. Launch Jupyter Notebook

```bash
jupyter notebook sms_spam_detection.ipynb
```

### 4. Run the Notebook

- **Option 1**: Click `Cell` → `Run All` to execute all cells sequentially
- **Option 2**: Press `Shift + Enter` to run cells one by one

> **Important**: Run cells in order from top to bottom, as each cell depends on previous ones.

---

## 📊 Dataset Information

- **Source:** [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Total Messages:** 5,574 (after removing duplicates)
- **Features:** 2 columns (label, message)
- **Class Distribution**:
  - **Ham (Legitimate)**: ~87% (4,827 messages)
  - **Spam**: ~13% (747 messages)
- **Encoding:** Latin-1

> **⚠️ Note:** Download the dataset from Kaggle and place `spam.csv` in this project folder before running the notebook.

### Sample Data

| Label | Message |
|-------|---------|
| ham   | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... |
| spam  | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 |

---

## 🔧 Technical Implementation

### 1. Data Preprocessing Pipeline

```python
# Steps performed on each message:
1. Convert to lowercase
2. Remove URLs
3. Remove special characters and digits
4. Remove extra whitespace
5. Remove stopwords (common words like "the", "is", "and")
6. Lemmatization (reduce words to base form)
```

**Example**:
```
Original: "WINNER!! You have won a $1000 prize. Call now!"
Cleaned:  "winner won prize call"
```

### 2. Feature Extraction (TF-IDF)

```python
TfidfVectorizer(
    max_features=3000,      # Top 3000 most important features
    ngram_range=(1, 2),     # Single words and word pairs
    min_df=2                # Word must appear in at least 2 documents
)
```

**Output**: 3,000-dimensional feature vector for each message

### 3. Machine Learning Models

#### Naive Bayes
```python
MultinomialNB()
```
- **Best for**: Text classification with word frequency
- **Advantage**: Fast training, works well with small datasets
- **Use case**: Baseline model for text classification

#### Logistic Regression
```python
LogisticRegression(max_iter=1000, random_state=42)
```
- **Best for**: Binary classification with probability estimates
- **Advantage**: Interpretable, provides confidence scores
- **Use case**: When you need probability of spam/ham

#### Support Vector Machine
```python
LinearSVC(max_iter=2000, random_state=42)
```
- **Best for**: High-dimensional text data
- **Advantage**: Robust to overfitting, excellent performance
- **Use case**: Production deployment for best accuracy

### 4. Train-Test Split

- **Training Set**: 80% (stratified sampling)
- **Test Set**: 20% (stratified sampling)
- **Random State**: 42 (for reproducibility)

---

## 📈 Model Performance

All three models achieve excellent results on the test set:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | >96% | >96% | >85% | >90% |
| Logistic Regression | >97% | >97% | >90% | >93% |
| SVM | >98% | >98% | >92% | >95% |

> **Note**: Exact scores may vary slightly due to random train-test split.

### Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of all predicted spam, how many were actually spam
- **Recall**: Of all actual spam, how many were detected
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)

---

## 🎨 Visualizations

The notebook includes comprehensive visualizations:

### 1. Data Exploration
- Class distribution (bar chart & pie chart)
- Message length distribution (spam vs ham)
- Word count distribution

### 2. Text Analysis
- **Word Clouds**: Visual representation of most frequent words
  - Green word cloud for legitimate messages
  - Red word cloud for spam messages

### 3. Model Evaluation
- **Confusion Matrices**: For each model (Naive Bayes, Logistic Regression, SVM)
- **Performance Comparison Charts**:
  - Individual metric bar charts (Accuracy, Precision, Recall, F1-Score)
  - Comprehensive grouped bar chart comparing all models

---

## 🧪 Interactive Testing

### Test Custom Messages

The notebook includes a `predict_spam()` function for testing:

```python
# Example usage
predict_spam("WINNER!! You have won a $1000 prize. Call now!", show_all_models=True)
```

**Output**:
```
================================================================================
SMS SPAM DETECTION RESULTS
================================================================================

Original Message: WINNER!! You have won a $1000 prize. Call now!
Cleaned Message:  winner won prize call

--------------------------------------------------------------------------------

📊 PREDICTIONS FROM ALL MODELS:

1. Naive Bayes:
   Prediction: SPAM
   Probability: Ham=0.0023, Spam=0.9977

2. Logistic Regression:
   Prediction: SPAM
   Probability: Ham=0.0015, Spam=0.9985

3. Support Vector Machine:
   Prediction: SPAM
   Decision Score: 2.4567 (negative=ham, positive=spam)

--------------------------------------------------------------------------------

🎯 CONSENSUS: 3/3 models predict SPAM
   ⚠️  This message is likely SPAM!

================================================================================
```

### Pre-loaded Test Examples

The notebook includes test cases for:

**Spam Messages**:
- "WINNER!! You have won a $1000 prize. Call now to claim your reward!"
- "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005"
- "Congratulations! You've been selected for a FREE iPhone. Click here now!"

**Legitimate Messages**:
- "Hey, are we still meeting for lunch tomorrow at 1pm?"
- "I'll be home in 10 minutes. Do you need anything from the store?"
- "Thanks for the birthday wishes! Had a great time at the party."

---

## 📚 Notebook Structure

The Jupyter Notebook is organized into 10 comprehensive sections:

1. **Import Libraries** - All required dependencies
2. **Load and Explore Dataset** - Data loading and initial exploration
3. **Exploratory Data Analysis (EDA)** - Statistical analysis and visualizations
4. **Text Preprocessing** - NLP pipeline for cleaning messages
5. **Feature Extraction (TF-IDF)** - Convert text to numerical features
6. **Model Training and Evaluation** - Train and evaluate all three models
   - 6.1 Naive Bayes Classifier
   - 6.2 Logistic Regression
   - 6.3 Support Vector Machine (SVM)
7. **Model Comparison** - Side-by-side performance analysis
8. **Interactive Testing** - Custom message prediction function
9. **Key Insights** - Automated analysis and findings
10. **Conclusion** - Summary and future improvements

---

## 🔍 Key Insights

### Common Spam Indicators

Words frequently found in spam messages:
- **Promotional**: "free", "win", "winner", "prize", "claim", "guaranteed"
- **Urgency**: "urgent", "now", "immediately", "limited time"
- **Actions**: "call", "text", "click", "reply"
- **Financial**: "$", "cash", "money", "reward"

### Message Characteristics

- **Spam messages** are typically longer (~138 characters on average)
- **Legitimate messages** are shorter (~71 characters on average)
- Spam messages often contain more capitalization and special characters
- Spam uses more promotional and action-oriented language

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'wordcloud'

**Solution**:
```bash
pip install wordcloud
```

See [INSTALL.md](INSTALL.md) for detailed troubleshooting.

#### 2. NameError: name 'X_train' is not defined

**Cause**: Running cells out of order

**Solution**: 
- Click `Kernel` → `Restart & Clear Output`
- Click `Cell` → `Run All`

#### 3. NLTK Data Not Found

**Solution**: The notebook automatically downloads required NLTK data. If issues persist:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Deep Learning Models**: LSTM, BERT, Transformer-based models
- [ ] **Additional Features**: 
  - Message metadata (timestamp, sender)
  - URL analysis and detection
  - Phone number pattern recognition
  - Special character frequency
- [ ] **Deployment Options**:
  - REST API with Flask/FastAPI
  - Web application interface
  - Mobile app integration
  - Browser extension
- [ ] **Advanced Techniques**:
  - Ensemble methods (voting, stacking)
  - Hyperparameter tuning with GridSearchCV
  - Cross-validation for robust evaluation
  - Active learning for continuous improvement
- [ ] **Multilingual Support**: Detect spam in multiple languages
- [ ] **Real-time Detection**: Stream processing for live SMS filtering

---

## 📖 Learning Resources

### Why These Algorithms?

**Naive Bayes**:
- Probabilistic classifier based on Bayes' theorem
- Assumes independence between features (words)
- Extremely fast and efficient for text classification
- Works well even with limited training data

**Logistic Regression**:
- Linear model for binary classification
- Provides probability estimates (confidence scores)
- Interpretable coefficients show word importance
- Good baseline for comparison

**Support Vector Machine (SVM)**:
- Finds optimal hyperplane to separate classes
- Excellent for high-dimensional data (like text)
- Robust to overfitting
- Often achieves best performance on text classification

### Why TF-IDF?

**TF-IDF** (Term Frequency-Inverse Document Frequency):
- **TF**: Measures how frequently a word appears in a document
- **IDF**: Reduces weight of common words, increases weight of rare words
- **Result**: Captures word importance across the entire dataset
- **Advantage**: Better than simple word counts for distinguishing spam from ham

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue describing the problem
2. **Suggest Features**: Share ideas for new functionality
3. **Improve Documentation**: Fix typos, add examples
4. **Add Models**: Implement new classification algorithms
5. **Optimize Performance**: Improve speed or accuracy

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Dataset**: UCI Machine Learning Repository - SMS Spam Collection
- **Libraries**: 
  - scikit-learn for machine learning tools
  - NLTK for natural language processing
  - pandas & numpy for data manipulation
  - matplotlib & seaborn for visualizations
  - wordcloud for text visualization

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [INSTALL.md](INSTALL.md) troubleshooting guide
2. Review the notebook comments and markdown cells
3. Ensure all cells are run in sequential order
4. Verify all dependencies are installed correctly

---

## 📊 Project Statistics

- **Total Code Cells**: 62
- **Total Lines of Code**: ~900+
- **Visualizations**: 10+ charts and plots
- **Models Implemented**: 3
- **Evaluation Metrics**: 4 per model
- **Test Examples**: 6 (3 spam, 3 ham)

---

## 🎓 Educational Value

This project is perfect for:
- **Students**: Learning machine learning and NLP concepts
- **Beginners**: Understanding text classification workflow
- **Practitioners**: Comparing different ML algorithms
- **Researchers**: Baseline for spam detection research

---

**Built with ❤️ using Python, scikit-learn, and Jupyter Notebook**

---

### 🌟 Star this project if you found it helpful!
