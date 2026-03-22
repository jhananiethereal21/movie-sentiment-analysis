from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from preprocess import load_and_preprocess

X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "../sentiment_model.pkl")
joblib.dump(vectorizer, "../vectorizer.pkl")
