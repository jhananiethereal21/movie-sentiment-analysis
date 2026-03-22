import joblib

def predict_sentiment(text):
    model = joblib.load("../sentiment_model.pkl")
    vectorizer = joblib.load("../vectorizer.pkl")
    vec = vectorizer.transform([text])
    return "Positive" if model.predict(vec)[0] == 1 else "Negative"

if __name__ == "__main__":
    print(predict_sentiment("This movie was fantastic!"))
