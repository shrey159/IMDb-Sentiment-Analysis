from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import joblib
import re
from pydantic import BaseModel


# Create FastAPI app
app = FastAPI()


# Allow frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Load trained model and preprocessing objects
model = joblib.load(BASE_DIR / "sentiment_model.pkl")
tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
chi_selector = joblib.load(BASE_DIR / "chi2_selector.pkl")


# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "IMDb Sentiment Analysis API is running"
    }


# Request format
class ReviewRequest(BaseModel):
    review: str


# Sentiment prediction endpoint
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):

    # Apply the same preprocessing used during training
    cleaned_review = preprocess_text(request.review)

    # Convert review into TF-IDF features
    review_tfidf = tfidf.transform([cleaned_review])

    # Apply Chi-Square feature selection
    review_selected = chi_selector.transform(review_tfidf)

    # Predict sentiment
    prediction = model.predict(review_selected)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "sentiment": sentiment
    }