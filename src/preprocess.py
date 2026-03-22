import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess(path="../data/IMDB-Dataset.csv"):
    df = pd.read_csv(path)
    X = df['review']
    y = df['sentiment'].map({'positive':1, 'negative':0})
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, vectorizer
