from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

import pickle

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# Load ML model
model = pickle.load(
    open("model/model.pkl", "rb")
)

vectorizer = pickle.load(
    open("model/vectorizer.pkl", "rb")
)

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data["text"]

    transformed = vectorizer.transform([text])

    prediction = model.predict(transformed)[0]

    confidence = max(
        model.predict_proba(transformed)[0]
    )

    return jsonify({
        "sentiment": prediction,
        "confidence": round(float(confidence), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)