from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import re

# ---------------------------
# Load models and preprocessors
# ---------------------------
tfidf = joblib.load('tfidf.pkl')
svd = joblib.load('svd.pkl')
clf = joblib.load('classifier.pkl')
reg = joblib.load('regressor.pkl')

try:
    numeric_transformer = joblib.load('numeric_transformer.pkl')
except:
    numeric_transformer = None

# ---------------------------
# Helper functions
# ---------------------------
MATH_SYMBOLS = r"[=<>+\-*/^]"
KEYWORDS = [
    "graph", "tree", "dp", "dynamic programming", "recursion",
    "greedy", "sort", "search", "matrix", "string"
]

def clean_text(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, list):
        x = " ".join(map(str, x))
    if isinstance(x, dict):
        x = " ".join([f"{k} {v}" for k, v in x.items()])
    x = str(x).lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def extract_numeric_features(text):
    features = {}
    features["text_length"] = len(text)
    features["math_symbol_count"] = len(re.findall(MATH_SYMBOLS, text))
    features["num_numbers"] = len(re.findall(r"\d+", text))

    sentences = [s for s in text.split(".") if s.strip()]
    if sentences:
        features["avg_sentence_len"] = np.mean(
            [len(s.split()) for s in sentences]
        )
    else:
        features["avg_sentence_len"] = 0

    for kw in KEYWORDS:
        features[f"kw_{kw}"] = int(kw in text)

    return features

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    desc = request.form.get("description", "")
    inp = request.form.get("input_description", "")
    out = request.form.get("output_description", "")

    full_text = clean_text(desc + " " + inp + " " + out)

    # Text features
    X_text = tfidf.transform([full_text])
    X_text_reduced = svd.transform(X_text)

    # Numeric features
    num_feats = extract_numeric_features(full_text)
    df_num = pd.DataFrame([num_feats])

    if numeric_transformer is not None:
        X_num = numeric_transformer.transform(df_num)
    else:
        X_num = df_num.values

    # Combine
    X_final = np.hstack([X_text_reduced, X_num])

    # Predictions
    class_pred = clf.predict(X_final)[0]
    score_pred = reg.predict(X_final)[0]

    class_map = {0: "Easy", 1: "Medium", 2: "Hard"}

    return render_template(
        "index.html",
        prediction_class=f"Predicted Difficulty Class: {class_map[class_pred]}",
        prediction_score=f"Predicted Difficulty Score: {round(float(score_pred), 2)}"
    )

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
