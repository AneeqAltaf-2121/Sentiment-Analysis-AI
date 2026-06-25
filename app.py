"""Command-line interface for the Sentiment Analysis AI project.

Handles user interaction only; all prediction logic lives in
`sentiment_model.py` so this file stays easy to swap for a Streamlit app,
FastAPI endpoint, or other future interface.
"""

from sentiment_model import predict_sentiment


def main():
    print("=" * 80)
    print("Sentiment Analysis AI")
    print("=" * 80)

    review = input("Enter a movie review: ")

    if not review.strip():
        print("\nPlease enter a non-empty review.")
        return

    result = predict_sentiment(review)

    if result["error"]:
        print(f"\nError: {result['error']}")
        return

    print(f"\nSentiment: {result['label']}")
    if result["confidence"] is not None:
        print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
