import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

st.title("📊 Simple Social Media Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'text' not in df.columns:
        st.error("❌ CSV must contain a 'text' column.")
    else:
        analyzer = SentimentIntensityAnalyzer()
        df['score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

        def label(score):
            if score >= 0.05:
                return 'Positive'
            elif score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'

        df['sentiment'] = df['score'].apply(label)

        st.subheader("Sentiment Count")
        st.bar_chart(df['sentiment'].value_counts())

        st.subheader("Data Preview")
        st.dataframe(df[['text', 'score', 'sentiment']])