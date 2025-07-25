# filepath: /nlp-misconceptions-predictor/nlp-misconceptions-predictor/scripts/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Here you can add any preprocessing steps if needed
    return data

def train_model(X_train, y_train):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)

def main():
    # Load the training data
    data = load_data('C:/Users/Admin/OneDrive/Documents/Comp_Project/nlp-misconceptions-predictor/data/train.csv')
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Split the data into features and target
    X = data['StudentExplanation']
    y = data['Category']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the model
    joblib.dump(model, 'C:/Users/Admin/OneDrive/Documents/Comp_Project/nlp-misconceptions-predictor/models/model.pkl')

if __name__ == "__main__":
    main()