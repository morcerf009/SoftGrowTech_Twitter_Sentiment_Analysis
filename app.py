import streamlit as st
import re
from textblob import TextBlob
from transformers import pipeline
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="X Sentiment Analyzer", layout="centered")

st.title("𝕏 Sentiment Analyzer")
st.write("Analyze the sentiment of posts from X (formerly Twitter) using NLP.")

# Load transformers model
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

transformer_model = load_model()


# Text preprocessing function
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = re.sub(r"#\w+", "", text)         # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters

    return text

# TextBlob sentiment
def analyze_textblob(text):

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.3:
        sentiment = "Positive"
    elif polarity < -0.3:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, polarity

# Emoji helper
def get_emoji(sentiment):

    if sentiment == "Positive":
        return "😊"
    elif sentiment == "Negative":
        return "😡"
    else:
        return "😐"

# Chart
def plot_sentiment(sentiment):

    data = {"Positive": 0, "Negative": 0, "Neutral": 0}
    data[sentiment] = 1

    fig, ax = plt.subplots()
    ax.bar(data.keys(), data.values())

    st.pyplot(fig)

# User input
user_input = st.text_area("What's happening?")

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Preprocess text
        clean_text = preprocess_text(user_input)

        st.subheader("Preprocessed Text")
        st.write(clean_text)

        # TextBlob analysis
        tb_sentiment, polarity = analyze_textblob(clean_text)

        st.subheader("TextBlob Engine")

        emoji = get_emoji(tb_sentiment)

        st.write("Sentiment:", tb_sentiment, emoji)
        st.write("Polarity Score:", round(polarity, 2))

        # Transformers analysis
        result = transformer_model(user_input)[0]

        st.subheader("Transformers Engine")

        label = result["label"]
        score = result["score"]

        st.write("Sentiment:", label)
        st.write("Confidence:", round(score, 2))

        # Chart
        st.subheader("Sentiment Visualization")
        plot_sentiment(tb_sentiment)