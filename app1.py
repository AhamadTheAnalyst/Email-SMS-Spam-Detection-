import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.set_page_config(layout="wide")

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #1f1c2c, #9287a9);
    color: white;
    padding: 20px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stTextArea, .stButton>button {
    border-radius: 10px;
    border: 2px solid #5a5a72;
    background-color: #3b3a4a;
    color: white;
}
.stButton>button:hover {
    background-color: #5a5a72;
    color: #f0f2f6;
    border-color: #f0f2f6;
}
.stHeader {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    color: #f0f2f6;
    margin-bottom: 20px;
    animation: fadeIn 2s;
}
.result-box {
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    font-size: 2em;
    font-weight: bold;
    margin-top: 30px;
    color: black;
    animation: slideIn 1s ease-out;
}
.spam {
    background-color: #ff6347; /* Tomato */
}
.not-spam {
    background-color: #3cb371; /* MediumSeaGreen */
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideIn {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

st.title("Email/SMS Spam Classifier 📧")
st.markdown("---")

input_sms = st.text_area("Enter the message you want to classify:", height=150)

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.markdown('<div class="result-box spam">Spam 🚫</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box not-spam">Not Spam ✅</div>', unsafe_allow_html=True)