from flask import Flask, request, jsonify
import pickle
from data_setup import remove_stopwords,data_cleaning  

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model files not found. Please train the model first.")


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Endpoint to predict sentiment from text
    
    Input JSON format: {"review_text": "your text here"}
    Output JSON format: {"sentiment_prediction": "positive" or "negative"}
    """
    try:
        # Get the input text
        data = request.get_json()
        if 'review_text' not in data:
            return jsonify({'error': 'No review_text field provided'}), 400
            
        text = data['review_text']
        # Text cleaning
        cleaned_text = data_cleaning(text)
        cleaned_text = remove_stopwords(text)
        # Vectorizing the text
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        return jsonify({
            'sentiment_prediction': 'positive' if prediction == 1 else 'negative'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500  #Flask defaults to returning 200 OK even when an error occurs hence the 500.

if __name__ == '__main__':
    app.run(debug=True, port=5002)