# SentimentIQ 🧠

A real-time sentiment analysis web app powered by Machine Learning (NLP). Paste any text and instantly get Positive, Negative, or Neutral sentiment with a confidence score.

## Features
- Real-time single text sentiment analysis
- Batch analysis for multiple texts at once
- Confidence score percentage
- REST API built with FastAPI
- Trained ML model using Scikit-learn + TF-IDF
- Interactive Jupyter Notebook for model training

## Tech Stack
- **ML / NLP:** Python, Scikit-learn, NLTK, Pandas, NumPy
- **Backend API:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, Vanilla JavaScript
- **Model:** Logistic Regression + TF-IDF Vectorizer
- **Notebook:** Jupyter Notebook

## Project Structure
```
sentimentiq/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── model.py         # ML model loader & predictor
│   └── requirements.txt
├── frontend/
│   └── index.html       # UI
├── model/
│   ├── sentiment_model.pkl   # Trained model (generated)
│   └── tfidf_vectorizer.pkl  # Vectorizer (generated)
├── notebook/
│   └── train_model.ipynb     # Model training notebook
└── README.md
```

## Getting Started

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the model
Open and run `notebook/train_model.ipynb` — this generates the `.pkl` files in `/model`.

### 3. Run the API
```bash
cd backend
uvicorn main:app --reload
```
API runs at `http://localhost:8000`

### 4. Open the frontend
Open `frontend/index.html` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Single text prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/docs` | Swagger UI |

## API Usage
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product, it is amazing!"}'
```

## Deployment
Deploy backend on **Render** (free) and frontend on **Vercel**.

## Author
Vasvi Shukla — [GitHub](https://github.com/VasviShukla) · [LinkedIn](https://www.linkedin.com/in/vasvishukla/)
