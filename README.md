# Email-SMS-Spam-Detection-

📧 Email/SMS Spam Classifier

# A machine learning-based web application built with Streamlit to classify Email/SMS messages as Spam 🚫 or Not Spam ✅.

The app uses Natural Language Processing (NLP) techniques for text preprocessing and a trained Machine Learning model (Multinomial Naive Bayes) to detect spam messages.

# 🚀 Features

Classifies messages into Spam or Not Spam

Preprocessing includes:

Lowercasing

Tokenization

Stopwords & punctuation removal

Stemming using PorterStemmer

Interactive and user-friendly Streamlit UI

Gradient background with styled buttons and animated results

# 📂 Project Structure

# 📦 Spam-Classifier
├── app1.py                # Streamlit app file

├── sms-spam-detection.ipynb # Jupyter notebook (model training & EDA)

├── spam.csv               # Dataset (SMS Spam Collection Dataset)

├── model.pkl              # Trained Machine Learning model

├── vectorizer.pkl         # TF-IDF Vectorizer

├── detect Spam.png        # Spam classification output screenshot

├── detect NOT Spam.png    # Not Spam classification output screenshot

└── README.md              # Project documentation



# Download NLTK resources

import nltk

nltk.download('punkt')

nltk.download('stopwords')


# Run the app
streamlit run app1.py

# 📊 Dataset

Dataset: SMS Spam Collection Dataset (spam.csv)

Contains 5,572 messages labeled as spam or ham (not spam).

Example:

Spam: "FREE subscription to Netflix for 1 year. Just provide your credit card details!"

Not Spam: "Good morning! Hope you have a great day 😊"

# 🧠 Model Training

The dataset was preprocessed using TF-IDF Vectorization.

Several Machine Learning algorithms were trained and evaluated, including:

# Logistic Regression

# Multinomial Naive Bayes

# Support Vector Machine (SVM)

# Random Forest Classifier

# Decision Tree Classifier

After comparing performance, Multinomial Naive Bayes was selected as the final model due to its:

High accuracy

Efficiency with text classification

Low computational cost

# ✨ Future Improvements

Add support for multiple languages

Deploy app using Streamlit Cloud / Heroku

Enhance model with deep learning (LSTMs, Transformers)

# 👨‍💻 Author

# Moh Ahamad

📧 Email: work.ahamad925@gmail.com

🔗 LinkedIn
