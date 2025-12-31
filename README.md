# AutoJudge: Predicting Programming Problem Difficulty

This project builds an intelligent system that automatically predicts the difficulty level and difficulty score of programming problems using only textual descriptions, similar to platforms like Codeforces, CodeChef, and Kattis.

The system performs:

Classification → Easy / Medium / Hard
Regression → Numerical difficulty score
Web-based inference using Flask + HTML

### 📌 Problem Statement

Online coding platforms classify problems based on difficulty, usually relying on:
Human judgment
User feedback and submissions
This project aims to automate difficulty estimation using:

Problem description
Input description
Output description

No metadata like submissions or tags is used — purely text-based prediction.

### 🚀 Features

Text preprocessing and cleaning
Feature engineering using:
TF-IDF vectors
Text length
Mathematical symbol count
Keyword frequencies (graph, dp, recursion, etc.)

Multiple ML models trained and compared
Best-performing models selected automatically
Unified preprocessing + model pipeline
Flask-based web interface for predictions

### 📂 Dataset

Each problem contains the following fields:

title
description
input_description
output_description
sample_io
problem_class   (Easy / Medium / Hard)
problem_score   (Numerical difficulty score)
url


### 🧪 Machine Learning Pipeline
🔹 Step 1: Data Preprocessing

Normalize text (handle lists, dicts, NaNs)
Combine all text fields into one full_text
Lowercasing and whitespace cleaning
Convert targets to numeric form

🔹 Step 2: Feature Engineering

Text-based features
TF-IDF (unigrams + bigrams)
Handcrafted features
Text length
Count of mathematical symbols (+ - * / % = < >)

Keyword frequency:
graph, tree, dp, recursion, greedy,
binary search, matrix, bfs, dfs, segment tree

🔹 Step 3: Model Training
Classification Models

Logistic Regression
Support Vector Machine (Linear SVM)
Random Forest Classifier
Regression Models
Linear Regression
Random Forest Regressor
Gradient Boosting Regressor

### ➡️ Best models are selected based on evaluation metrics.

🔹 Step 4: Evaluation

Classification
Accuracy
Confusion Matrix
Precision / Recall / F1-score

Regression

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)

🔹 Step 5: Model Saving

All preprocessing and models are saved using joblib:

pipeline_classifier.pkl
pipeline_regressor.pkl

This ensures consistent preprocessing during inference.

### 🌐 Web Interface (Flask)
Input Fields

Problem Description
Input Description
Output Description

Output

Predicted Difficulty Class (Easy / Medium / Hard)
Predicted Difficulty Score

### 🗂️ Project Structure
Autojudge/

│
├── app.py                  # Flask backend
├── custom_transformers.py  # Feature engineering transformers
├── templates/
│   └── index.html          # HTML frontend
│
├── pipeline_classifier.pkl # Saved classification pipeline
├── pipeline_regressor.pkl  # Saved regression pipeline
│
├── requirements.txt
├── README.md

### ▶️ How to Run Locally
1️⃣ Create virtual environment
python -m venv myenv
myenv\Scripts\activate   # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Flask app
python app.py

4️⃣ Open browser
http://127.0.0.1:5000/

### 📦 Requirements

Key libraries used:
Python 3.10+
scikit-learn
pandas
numpy
scipy
flask
joblib
(Exact versions pinned in requirements.txt)

### 📈 Results & Observations

The classification model achieved an overall accuracy of ~53% using only textual information.
Performance is strongest for Hard problems, with high recall (0.87), indicating the model reliably identifies difficult problems.
Easy and Medium classes are harder to distinguish, due to overlapping language patterns and subjective difficulty definitions.
The regression model achieved:
MAE: 1.60
RMSE: 1.92
Predicting an exact difficulty score is challenging, as problem difficulty is inherently subjective and noisy.
Feature engineering (keyword frequency, text length, mathematical symbols) significantly improves performance over raw TF-IDF alone.

### 👨‍💻 Author

Rishabh Singhal
23113127
Project: Automatic Programming Problem Difficulty Prediction
