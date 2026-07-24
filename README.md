# 📰 Fake News Detection using Machine Learning and NLP

## 📌 Project Overview
This project is a Machine Learning-based web application that predicts whether a news article is **Fake** or **Real**. The model is trained on a labeled dataset using Natural Language Processing (NLP) techniques and deployed using Streamlit.

## ✨ Features

- Detects whether a news article is **Real** or **Fake**
- Performs text preprocessing using NLP techniques
- Uses TF-IDF vectorization for feature extraction
- Compares five machine learning models:
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - Neural Network (MLP)
- Evaluates models using accuracy, confusion matrix, and classification report
- Provides predictions through an interactive Streamlit web application

## 📂 Dataset

This project uses the **Fake News Dataset** from Kaggle to classify news articles as **Real** or **Fake**.

### Dataset Information
- **Source:** Kaggle
- **Features:** Title, Text, Label
- **Classes:** Real News and Fake News

### Preprocessing
- Converted text to lowercase
- Removed URLs and HTML tags
- Removed punctuation
- Removed stopwords
- Applied lemmatization
- Performed TF-IDF vectorization

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Joblib

## 🤖 Machine Learning Models Compared

The following machine learning models were trained and evaluated for fake news classification:

| Model | Accuracy |
|--------|----------|
| K-Nearest Neighbors (KNN) | 71.85% |
| Naive Bayes | 93.01% |
| Logistic Regression | 98.40% |
| Random Forest | 99.71% |
| Neural Network (MLP) | 99.02% |

> **Note:** Although the Random Forest model achieved the highest test accuracy (99.71%), the Naive Bayes model demonstrated better generalization on manually tested unseen news articles. Therefore, the Naive Bayes model was selected for deployment in the Streamlit application.

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
```

2. Navigate to the project directory:
```bash
cd Fake-News-Detection
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ How to Run

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your default web browser, where you can enter a news article and receive a prediction indicating whether it is **Real** or **Fake**.

## 📁 Project Structure

Fake-News-Detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── images/
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── Fake_News_Detection.ipynb
├── reports/
└── src/

## 📸 Screenshots

### Streamlit Web Application

*(Insert your Streamlit app screenshot here.)*

### Model Performance Comparison

*(Insert the accuracy comparison graph here.)*

## 🚀 Future Improvements

- Improve prediction accuracy using transformer-based models such as BERT.
- Deploy the application on a cloud platform.
- Support multilingual fake news detection.
- Enable real-time news verification using online APIs.

## 👤 Author

**Ashitha Dizousa**

Summer Internship Project – AI & Machine Learning