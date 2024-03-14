import numpy as np
from flask import Flask, request, render_template
import pickle
from emotion_analyzer import analyze_emotion

app = Flask(__name__)

# Load the model and vectorizer
with open("model.pkl", "rb") as model_file:
    loaded_svm_classifier = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    loaded_tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route("/")
def home():
    return render_template("index.html", prediction=None, user_input=None)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        user_input = request.form['text_input']

        # Use the loaded model and vectorizer
        cleaned_text = analyze_emotion(user_input)
        tfidf_text = loaded_tfidf_vectorizer.transform([cleaned_text])
        prediction = loaded_svm_classifier.predict(tfidf_text)[0]

        return render_template('index.html', prediction=prediction, user_input=user_input, predicted_value=prediction)

if __name__ == '__main__':
    app.run(debug=True)
