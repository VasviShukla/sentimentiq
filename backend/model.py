import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data on first run
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'tfidf_vectorizer.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    """Clean and preprocess input text."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()           # clean whitespace
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def load_model():
    """Load trained model and vectorizer."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Please run the Jupyter notebook (notebook/train_model.ipynb) first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_sentiment(text: str, model, vectorizer) -> dict:
    """Predict sentiment for a single text."""
    cleaned = preprocess(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = label_map.get(prediction, str(prediction))

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            "negative": round(float(probabilities[0]) * 100, 2),
            "neutral": round(float(probabilities[1]) * 100, 2),
            "positive": round(float(probabilities[2]) * 100, 2),
        }
    }
