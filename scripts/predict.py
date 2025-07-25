# filepath: /nlp-misconceptions-predictor/nlp-misconceptions-predictor/scripts/predict.py

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def load_model(model_path):
    """Load the trained model from the specified path."""
    model = joblib.load(model_path)
    return model

def preprocess_data(test_data):
    """Preprocess the test data for prediction."""
    # Assuming the test data has a 'StudentExplanation' column
    return test_data['StudentExplanation']

def make_predictions(model, test_data):
    """Make predictions on the test data using the loaded model."""
    preprocessed_data = preprocess_data(test_data)
    predictions = model.predict(preprocessed_data)
    return predictions

def main():
    # Load the test data
    test_data_path = '../data/test.csv'
    test_data = pd.read_csv(test_data_path)

    # Load the trained model
    model_path = '../models/model.pkl'
    model = load_model(model_path)

    # Make predictions
    predictions = make_predictions(model, test_data)

    # Prepare the submission DataFrame
    submission = pd.DataFrame({
        'row_id': test_data['row_id'],
        'Category:Misconception': predictions
    })

    # Save the submission file
    submission_path = '../data/sample_submission.csv'
    submission.to_csv(submission_path, index=False)

if __name__ == "__main__":
    main()