import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import nltk
import re

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        return ""

# Modularize the model creation
def create_model(input_dim):
    try:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error during model creation: {e}")
        return None

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    try:
        loss, accuracy = model.evaluate(X_test, y_test)
        predictions = model.predict(X_test, verbose=0)
        predicted_labels = (predictions >= 0.5).astype(int).ravel()

        precision = precision_score(y_test, predicted_labels, zero_division=0)
        recall = recall_score(y_test, predicted_labels, zero_division=0)
        f1 = f1_score(y_test, predicted_labels, zero_division=0)

        return accuracy, precision, recall, f1
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None, None, None, None

# Load dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'feedback' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'feedback' and 'label' columns.")

        df['cleaned'] = df['feedback'].apply(preprocess_text)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Main pipeline
def main():
    try:
        df = load_dataset("feedback.csv")
        if df is None:
            return

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['cleaned']).toarray()
        y = np.array(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = create_model(X.shape[1])
        if model is None:
            return

        model.fit(X_train, y_train, epochs=20, batch_size=4, verbose=1)

        evaluate_model(model, X_test, y_test)

        while True:
            user_input = input("Enter customer feedback (or type 'exit'): ")
            if user_input.lower() == 'exit':
                break
            result = predict_feedback(user_input, vectorizer, model)
            print(f"Sentiment: {result}")
    except Exception as e:
        print(f"Error in main pipeline: {e}")

def predict_feedback(text, vectorizer, model):
    try:
        cleaned = preprocess_text(text)
        if not cleaned:
            return "Invalid input, unable to process."
        vector = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vector, verbose=0)
        return "Positive" if prediction[0][0] >= 0.5 else "Negative"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction"

if __name__ == "__main__":
    main()
