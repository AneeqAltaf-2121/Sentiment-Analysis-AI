from fastapi import FastAPI
from pydantic import BaseModel

from sentiment_model import predict_sentiment


app = FastAPI(
    title="Sentiment Analysis AI API",
    description="API for predicting whether a movie review is positive or negative.",
    version="1.0.0",
)


class ReviewRequest(BaseModel):
    review: str


@app.get("/")
def home():
    return {"message": "Sentiment Analysis AI API is running."}


@app.post("/predict")
def predict(request: ReviewRequest):
    result = predict_sentiment(request.review)

    if result["error"]:
        return {
            "success": False,
            "error": result["error"],
        }

    return {
        "success": True,
        "sentiment": result["label"],
        "confidence": result["confidence"],
    }