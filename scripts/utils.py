# filepath: /nlp-misconceptions-predictor/nlp-misconceptions-predictor/scripts/utils.py

import pandas as pd
import re
import numpy as np
from sklearn.metrics import classification_report

def load_data(filepath):
    """Load CSV data from the specified filepath."""
    return pd.read_csv(filepath)

def clean_text(text):
    """Clean and preprocess the input text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def preprocess_data(df):
    """Apply text cleaning to the 'StudentExplanation' column."""
    df['CleanedExplanation'] = df['StudentExplanation'].apply(clean_text)
    return df

def evaluate_model(y_true, y_pred):
    """Evaluate the model's performance using classification metrics."""
    return classification_report(y_true, y_pred, target_names=['True_Correct', 'False_Neither', 'False_Misconception'])