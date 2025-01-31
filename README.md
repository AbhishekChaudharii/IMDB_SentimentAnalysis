# Sentiment Analysis Pipeline

## Overview
This project is a sentiment analysis API built using Flask. It allows users to analyze the sentiment of movie reviews as either "positive" or "negative" using a trained Logistic Regression or Naive Bayes model.


## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install -r requirements.txt
```


## How to use this?

### 1. Prepare Data
Run the following command to clean and store the IMDB dataset in an SQLite database:
```bash
python data_setup.py
```

### 2. Train the Model
Train the sentiment analysis model and export the trained model and vectorizer:
```bash
python train_model.py
```

### 3. Run the API
Start the Flask server:
```bash
python app.py
```

### 4. Predict Sentiment
Send a POST request to `/predict` with JSON input:
```bash
curl -X POST "http://127.0.0.1:5002/predict" \
     -H "Content-Type: application/json" \
     -d '{"review_text": "This movie was amazing!"}'
```
#### Example Response
```json
{
    "sentiment_prediction": "positive"
}
```

## Notes
- Ensure `sentiment_model.pkl` and `vectorizer.pkl` exist before running the API. If not, train the model first.
- The dataset should be placed in the `Data` folder with the filename `IMDB_Dataset.csv`.



