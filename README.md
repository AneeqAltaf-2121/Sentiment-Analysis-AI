# Sentiment Analysis AI

Sentiment Analysis AI is a Natural Language Processing (NLP) project built with Python, Pandas, NLTK, Scikit-Learn, and FastAPI. The system classifies movie reviews as **Positive** or **Negative** by combining text preprocessing, TF-IDF feature extraction, and machine learning models trained on the IMDb movie reviews dataset.

---

## Features

- Predicts whether a movie review is **Positive** or **Negative**
- Cleans and preprocesses raw text using NLTK
- Removes HTML tags, URLs, punctuation, numbers, and stopwords
- Applies lemmatization for better text normalization
- Converts text into numerical features using **TF-IDF**
- Trains and evaluates multiple machine learning models
- Compares Logistic Regression and Multinomial Naive Bayes
- Automatically selects and saves the best-performing model
- Command-line interface for real-time sentiment prediction
- FastAPI REST API for real-time sentiment prediction
- Modular project structure for future web or GUI deployment

---

## Project Structure

```
Sentiment-Analysis-AI/
│
├── data/
│   ├── IMDB Dataset.csv
│
├── models/
│
├── notebooks/
│   ├── data_exploration.py
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   └── model_comparison.py
│
├── api.py
├── sentiment_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-Learn
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes
- FastAPI
- Uvicorn
- Joblib
- Git
- GitHub

---

## Dataset

This project uses the **IMDb Movie Reviews Dataset**, containing **50,000** labeled movie reviews evenly divided between positive and negative sentiments.

**Dataset Source:**

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Dataset includes:

- 50,000 movie reviews
- Binary sentiment labels
- Balanced class distribution

---

## Machine Learning Pipeline

```
IMDb Reviews
      │
      ▼
Data Exploration
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Feature Extraction
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
(Logistic Regression &
Multinomial Naive Bayes)
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
Real-Time Prediction
(CLI & FastAPI)
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/AneeqAltaf-2121/Sentiment-Analysis-AI.git
```

Navigate into the project

```bash
cd Sentiment-Analysis-AI
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application

```bash
python app.py
```

Enter a movie review when prompted.

Example:

```text
Enter a movie review:

This movie was absolutely fantastic. The acting was brilliant and the story kept me engaged from beginning to end.

Sentiment: Positive
Confidence: 97.42%
```

---

## Example Output

### Positive Review

<img width="1464" height="130" alt="positive-review" src="https://github.com/user-attachments/assets/85b5f4e8-a23c-4b2c-b233-3b1d3c242010" />

---

### Negative Review

<img width="1472" height="170" alt="negative-review" src="https://github.com/user-attachments/assets/8f11cf5f-aec0-44c6-a95f-2f2a56f91050" />

---

## Model Evaluation

The project compares two classical machine learning algorithms:

- Logistic Regression
- Multinomial Naive Bayes

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

The best-performing model is automatically selected and saved for prediction.

---

## Skills Demonstrated

- Natural Language Processing (NLP)
- Text Preprocessing
- Feature Engineering
- TF-IDF Vectorization
- Machine Learning
- Model Evaluation
- Model Comparison
- Python Programming
- Scikit-Learn
- NLTK
- Git & GitHub
- FastAPI
- REST API Development

---


## FastAPI REST API

Run the API server:

```bash
uvicorn api:app --reload
```

Once the server is running, open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The Swagger UI allows you to test the sentiment prediction endpoint directly from your browser.

Example request:

```json
{
  "review": "This movie was absolutely fantastic. The acting was brilliant, the story was engaging, and I enjoyed every minute of it. I would definitely recommend it to others."
}
```

Example response:

```json
{
  "success": true,
  "sentiment": "Positive",
  "confidence": 0.9892861642774675
}
```

---

## Future Improvements

- Streamlit web interface
- Transformer-based sentiment analysis using BERT
- Docker deployment
- Cloud deployment (Render / Railway / Azure)

---

## Author

**Aneeq Altaf**

GitHub Profile: https://github.com/AneeqAltaf-2121
