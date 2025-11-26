# Movie Genre Classification

A machine learning project that predicts movie genres based on plot summaries using TF-IDF vectorization and multiple classification algorithms.

## 📋 Project Overview

This project implements a text classification system that can predict the genre of a movie based on its plot description. The model uses natural language processing techniques and compares three different machine learning algorithms:

- **Naive Bayes Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

## 📁 Dataset

**Source:** [Genre Classification Dataset - IMDB (Kaggle)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

> **⚠️ Note:** The dataset files are NOT included in the GitHub repository due to their large size. Please download them from the Kaggle link above.

The dataset is located in the `Genre Classification Dataset` folder:
- **train_data.txt**: ~54,000 movies with ID, Title, Genre, and Description
- **test_data.txt**: ~54,000 movies with ID, Title, and Description (no genre labels)
- **description.txt**: Information about the dataset format

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed. Install the required packages:

```bash
pip install -r requirements.txt
```

```

> **⚠️ IMPORTANT: Dataset Download Required**
> The dataset files are NOT included in this repository because they are large.
> 
> **You must download them from Kaggle:** [Click Here to Download](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
> 
> After downloading, extract the files into the `Genre Classification Dataset` folder.

### Running the Notebook

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to and open `movie_genre_classifier.ipynb`

3. Run all cells in order (Cell → Run All)

## 📊 What the Notebook Does

1. **Data Loading**: Reads and parses the training and test datasets
2. **Exploratory Data Analysis**: 
   - Analyzes genre distribution
   - Visualizes text length statistics
   - Identifies data patterns

3. **Text Preprocessing**:
   - Converts text to lowercase
   - Removes special characters and digits
   - Cleans whitespace

4. **Feature Engineering**:
   - TF-IDF vectorization with 5000 features
   - Bigram support (1-2 word combinations)

5. **Model Training & Evaluation**:
   - Trains three different classifiers
   - Compares performance metrics
   - Generates detailed classification reports
   - Creates confusion matrices

6. **Predictions**:
   - Predicts genres for test dataset
   - Saves results to `predictions.txt`

## 📈 Expected Results

The models typically achieve:
- **Validation Accuracy**: 50-65% (depending on the model)
- **Best Performer**: Usually Logistic Regression or SVM

Note: Genre classification is a challenging task due to:
- Multiple possible genres per movie
- Subjective genre boundaries
- Imbalanced dataset (some genres are much more common)

## 📝 Output Files

- **predictions.txt**: Contains predicted genres for test data in format: `ID ::: PREDICTED_GENRE`

## 🛠️ Techniques Used

- **Text Preprocessing**: Regular expressions, lowercasing
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classification Algorithms**:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear Support Vector Classification
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## 📚 Libraries Used

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning algorithms and tools
- **matplotlib & seaborn**: Data visualization

## 🎯 Future Improvements

- Experiment with word embeddings (Word2Vec, GloVe)
- Try deep learning models (LSTM, BERT)
- Handle multi-label classification (movies with multiple genres)
- Implement cross-validation for more robust evaluation
- Hyperparameter tuning using GridSearchCV

## 📄 License

This project is for educational purposes.

## 👤 Author

Created as part of the CodSoft internship program.

---

**Happy Coding! 🎬🤖**
