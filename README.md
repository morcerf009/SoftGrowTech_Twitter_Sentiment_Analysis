# 𝕏 Sentiment Analysis (formerly Twitter)

This project analyzes the sentiment of posts on 𝕏 (formerly known as Twitter) using Natural Language Processing.

## Technologies Used
- Python
- TextBlob
- NLTK
- Hugging Face Transformers
- Streamlit

## Features
- **Dynamic UI**: Modern web interface built with Streamlit.
- **Advanced Model**: Uses Hugging Face Transformers for high-accuracy sentiment analysis.
- **Text Pre-processing**: Automatically cleans URLs and mentions for better results.
- **Dual Analysis**: Compares TextBlob and RoBERTa models in real-time.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
streamlit run app.py
```

## Example Output

| Input | Sentiment | Model |
|-------|-----------|-------|
| I love this update | Positive | TextBlob |
| This change is confusing | Negative | Transformers |
| 𝕏 is evolving | Neutral | Both |