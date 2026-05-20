import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.preprocess import clean_text

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove null rows
data.dropna(inplace=True)

# Clean text
data["review"] = data["review"].apply(clean_text)

# Features and labels
X = data["review"]
y = data["sentiment"]

# Better vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_vectorized = vectorizer.fit_transform(X)

# Better train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

# Better model
model = LogisticRegression(
    max_iter=2000
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Save model
pickle.dump(
    model,
    open("model/model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("model/vectorizer.pkl", "wb")
)

print("Training completed!")