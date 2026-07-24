import streamlit as st
import joblib
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)
st.title("📰 Fake News Detection")
st.sidebar.title("About")

st.sidebar.info(
    """
    This application uses a Machine Learning model
    trained on thousands of news articles to classify
    news as **Real** or **Fake**.
    """
)
st.markdown(
    """
    Enter or paste a news article below and click **Predict**.
    The model will analyze the text and classify it as **Real** or **Fake**.
    """
)
user_input = st.text_area(
    "Paste the news article here:",
    height=250
)
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a news article.")

    else:
        cleaned_text = clean_text(user_input)

        vectorized_text = vectorizer.transform([cleaned_text])

        

        prediction = model.predict(vectorized_text)
        probability = model.predict_proba(vectorized_text)

        confidence = probability.max() * 100
        if prediction[0] == 0:
           st.error("🚨 Prediction: FAKE NEWS")
           st.write(f"**Confidence:** {confidence:.2f}%")
           st.write("⚠️ This article appears to contain misleading or false information.")
        else:
           st.success("✅ Prediction: REAL NEWS")
           st.write(f"**Confidence:** {confidence:.2f}%")
           st.write("✔️ This article appears to be from a reliable news source.") 