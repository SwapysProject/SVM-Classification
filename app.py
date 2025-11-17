import pickle
import streamlit as st
import scipy.sparse as sp
import numpy as np
import pandas as pd
import re
import string
from textblob import TextBlob

# Load the saved model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('best_baseline_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# TextBlob sentiment analysis
def get_textblob_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return 'positive', polarity
    elif polarity < -0.1:
        return 'negative', polarity
    else:
        return 'neutral', polarity

# Streamlit UI
st.title("🎭 Sentiment Analysis App")
st.write("Enter text to analyze its sentiment using our trained SVM model.")

# Main text input
user_input = st.text_area(
    "Enter your text here:",
    height=150,
    placeholder="Type or paste your text here... (e.g., tweets, reviews, comments)"
)

# Prediction button
if st.button("🔍 Analyze Sentiment", type="primary"):
    if user_input.strip():
        # Clean the input text
        cleaned_input = clean_text(user_input)
        
        # ===== Our SVM Model Prediction =====
        input_vectorized = vectorizer.transform([cleaned_input])
        svm_prediction = model.predict(input_vectorized)[0]
        svm_prediction_proba = model.predict_proba(input_vectorized)[0]
        svm_confidence = max(svm_prediction_proba) * 100
        
        # ===== TextBlob Prediction =====
        textblob_sentiment, textblob_polarity = get_textblob_sentiment(user_input)
        textblob_confidence = abs(textblob_polarity) * 100
        
        # Display results side by side
        st.subheader("Comparison Results:")
        
        col1, col2 = st.columns(2)
        
        # Our Model Results
        with col1:
            st.markdown("### 🤖 Our SVM Model")
            if svm_prediction == 'positive':
                st.success(f"**Sentiment: {svm_prediction.upper()} 😊**")
            elif svm_prediction == 'negative':
                st.error(f"**Sentiment: {svm_prediction.upper()} 😞**")
            else:
                st.info(f"**Sentiment: {svm_prediction.upper()} 😐**")
            
            st.metric("Confidence", f"{svm_confidence:.2f}%")
            
            # Show probability distribution for SVM
            st.write("**Probability Distribution:**")
            svm_proba_dict = {label: prob for label, prob in zip(model.classes_, svm_prediction_proba)}
            st.bar_chart(svm_proba_dict)
        
        # TextBlob Results
        with col2:
            st.markdown("### 📚 TextBlob Model")
            if textblob_sentiment == 'positive':
                st.success(f"**Sentiment: {textblob_sentiment.upper()} 😊**")
            elif textblob_sentiment == 'negative':
                st.error(f"**Sentiment: {textblob_sentiment.upper()} 😞**")
            else:
                st.info(f"**Sentiment: {textblob_sentiment.upper()} 😐**")
            
            st.metric("Polarity Score", f"{textblob_polarity:.3f}")
            st.caption(f"Confidence: ~{textblob_confidence:.2f}%")
            
            # Show polarity bar
            st.write("**Polarity Range:**")
            polarity_df = pd.DataFrame({
                'negative': [-1, textblob_polarity] if textblob_polarity < 0 else [0, 0],
                'neutral': [0, 0],
                'positive': [1, textblob_polarity] if textblob_polarity > 0 else [0, 0]
            })
            st.bar_chart(polarity_df)
        
        # Comparison Summary
        st.divider()
        st.subheader("📊 Comparison Summary")
        
        agreement = "✅ Both models agree" if svm_prediction == textblob_sentiment else "⚠️ Models disagree"
        st.write(f"**{agreement}**")
        
        comparison_df = pd.DataFrame({
            'Model': ['Our SVM', 'TextBlob'],
            'Prediction': [svm_prediction, textblob_sentiment],
            'Confidence/Score': [f"{svm_confidence:.2f}%", f"{textblob_polarity:.3f}"]
        })
        st.table(comparison_df)
        
        # Show input details
        with st.expander("View Input Details"):
            st.write(f"**Original Text:** {user_input}")
            st.write(f"**Cleaned Text:** {cleaned_input}")
        
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Support Vector Machine (SVM) model "
    "trained on sentiment analysis data to predict whether "
    "text expresses positive, negative, or neutral sentiment."
)

st.sidebar.header("How to use")
st.sidebar.markdown("""
1. Enter or paste your text in the text area
2. Click 'Analyze Sentiment' button
3. View the predicted sentiment and confidence score
""")