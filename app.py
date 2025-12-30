from flask import Flask, request, render_template
import joblib
import pandas as pd

#  VERY IMPORTANT: import custom transformers
from custom_transformers import HandcraftedFeatures, FeatureUnion

# Load models
clf_model = joblib.load("difficulty_classifier.pkl")
reg_model = joblib.load("difficulty_regressor.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form inputs
    title = request.form.get("title", "")
    description = request.form.get("description", "")
    input_desc = request.form.get("input_description", "")
    output_desc = request.form.get("output_description", "")
    sample_io = request.form.get("sample_io", "")

    # Combine text exactly as training
    full_text = " ".join([
        title,
        description,
        input_desc,
        output_desc,
        sample_io
    ])

    X = pd.Series([full_text])

    # Predictions
    pred_class = clf_model.predict(X)[0]
    pred_score = reg_model.predict(X)[0]

    return render_template(
        "index.html",
        predicted_class=pred_class,
        predicted_score=round(pred_score, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
