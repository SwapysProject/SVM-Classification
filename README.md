# Sentiment Analysis App

A sentiment analysis application that compares a trained SVM model with TextBlob for text sentiment classification.

## Features

- **SVM Model**: Custom-trained Support Vector Machine on sentiment data
- **TextBlob Comparison**: Side-by-side comparison with TextBlob pre-trained model
- **Interactive UI**: Streamlit-based web interface
- **Real-time Predictions**: Instant sentiment analysis with confidence scores

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Sentiment-Analysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## Project Structure

```
├── app.py                    # Streamlit web application
├── eda.ipynb                 # Exploratory data analysis & model training
├── tuning.ipynb              # Hyperparameter tuning
├── train.csv                 # Training dataset
├── requirements.txt          # Python dependencies
├── best_baseline_model.pkl   # Trained SVM model
├── vectorizer.pkl            # TF-IDF vectorizer
└── README.md                 # Documentation
```

## Models

### SVM Model
- **Vectorization**: TF-IDF with 2000 max features, unigrams + bigrams
- **Algorithm**: Support Vector Machine with linear kernel
- **Training**: Trained on 27K+ labeled text samples

### TextBlob
- Pre-trained sentiment analyzer
- Polarity score range: -1 (negative) to +1 (positive)

## Dataset

The training data includes:
- **Text samples**: 27,483 entries
- **Sentiment labels**: positive, negative, neutral
- **Additional features**: Time of Tweet, Age of User, Country demographics

## Performance

Check `baseline_results.csv` for model comparison metrics (Accuracy & F1 Score).

## Requirements

- Python 3.11+
- streamlit
- scikit-learn
- textblob
- pandas
- numpy
- scipy


