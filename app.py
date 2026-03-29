from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# -------------------------------
# Train model using dummy data
# -------------------------------
def train_model():
    np.random.seed(42)
    data_size = 500

    sleep_hours = np.random.uniform(3, 10, data_size)
    screen_time = np.random.uniform(1, 12, data_size)
    mood_score = np.random.randint(1, 11, data_size)          # 1 to 10
    activity_level = np.random.randint(1, 11, data_size)      # 1 to 10
    questionnaire_score = np.random.randint(5, 31, data_size) # 5 to 30

    stress_level = []

    for i in range(data_size):
        score = 0

        if sleep_hours[i] < 5:
            score += 2
        elif sleep_hours[i] < 7:
            score += 1

        if screen_time[i] > 8:
            score += 2
        elif screen_time[i] > 5:
            score += 1

        if mood_score[i] <= 3:
            score += 2
        elif mood_score[i] <= 6:
            score += 1

        if activity_level[i] <= 3:
            score += 2
        elif activity_level[i] <= 6:
            score += 1

        if questionnaire_score[i] > 22:
            score += 2
        elif questionnaire_score[i] > 15:
            score += 1

        if score <= 3:
            stress_level.append("Low")
        elif score <= 6:
            stress_level.append("Medium")
        else:
            stress_level.append("High")

    df = pd.DataFrame({
        "sleep_hours": sleep_hours,
        "screen_time": screen_time,
        "mood_score": mood_score,
        "activity_level": activity_level,
        "questionnaire_score": questionnaire_score,
        "stress_level": stress_level
    })

    X = df[["sleep_hours", "screen_time", "mood_score", "activity_level", "questionnaire_score"]]
    y = df["stress_level"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


model = train_model()

# -------------------------------
# Recommendation logic
# -------------------------------
def get_recommendation(prediction):
    if prediction == "Low":
        return [
            "Maintain your current routine.",
            "Keep a healthy sleep schedule.",
            "Continue regular physical activity.",
            "Take short breaks during work."
        ]
    elif prediction == "Medium":
        return [
            "Try reducing screen time before sleep.",
            "Practice meditation or deep breathing for 10 to 15 minutes daily.",
            "Improve sleep quality and keep a fixed bedtime.",
            "Take regular breaks and talk to friends or family."
        ]
    else:
        return [
            "Consider speaking with a mental health professional or counselor.",
            "Reduce stress triggers and limit excessive screen time.",
            "Focus on proper sleep, hydration, and daily movement.",
            "Practice mindfulness and reach out for emotional support."
        ]

# -------------------------------
# HTML template
# -------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Prediction System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f9;
            margin: 0;
            padding: 0;
        }

        .container {
            width: 90%;
            max-width: 700px;
            margin: 40px auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(0,0,0,0.12);
        }

        h1 {
            text-align: center;
            color: #222;
            margin-bottom: 25px;
        }

        label {
            font-weight: bold;
            display: block;
            margin-top: 15px;
            margin-bottom: 6px;
        }

        input {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-size: 15px;
            box-sizing: border-box;
        }

        button {
            margin-top: 20px;
            width: 100%;
            padding: 12px;
            background: #007bff;
            border: none;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
        }

        button:hover {
            background: #0056b3;
        }

        .result {
            margin-top: 25px;
            padding: 15px;
            border-radius: 10px;
            background: #eef7ee;
            border-left: 5px solid #28a745;
        }

        .error {
            margin-top: 20px;
            padding: 12px;
            border-radius: 8px;
            background: #fdeaea;
            color: #b30000;
            border-left: 5px solid #dc3545;
        }

        ul {
            margin-top: 10px;
        }

        .note {
            margin-top: 20px;
            font-size: 14px;
            color: #666;
        }

        .footer {
            margin-top: 20px;
            text-align: center;
            color: #666;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Mental Health Prediction + Recommendation System</h1>

        <form method="POST">
            <label for="sleep_hours">Sleep Hours (per day)</label>
            <input type="number" step="0.1" name="sleep_hours" id="sleep_hours" required>

            <label for="screen_time">Screen Time (hours per day)</label>
            <input type="number" step="0.1" name="screen_time" id="screen_time" required>

            <label for="mood_score">Mood Score (1 to 10)</label>
            <input type="number" name="mood_score" id="mood_score" min="1" max="10" required>

            <label for="activity_level">Activity Level (1 to 10)</label>
            <input type="number" name="activity_level" id="activity_level" min="1" max="10" required>

            <label for="questionnaire_score">Questionnaire Score (5 to 30)</label>
            <input type="number" name="questionnaire_score" id="questionnaire_score" min="5" max="30" required>

            <button type="submit">Predict Stress Level</button>
        </form>

        {% if error %}
            <div class="error">
                <strong>Error:</strong> {{ error }}
            </div>
        {% endif %}

        {% if prediction %}
            <div class="result">
                <h2>Predicted Stress Level: {{ prediction }}</h2>
                <h3>Recommended Suggestions:</h3>
                <ul>
                    {% for item in recommendations %}
                        <li>{{ item }}</li>
                    {% endfor %}
                </ul>
            </div>
        {% endif %}

        <p class="note">
            Note: This project is for educational purposes only and is not a medical diagnosis tool.
        </p>

        <div class="footer">
            Built with Python, Flask, Pandas, NumPy, and Scikit-learn
        </div>
    </div>
</body>
</html>
"""

# -------------------------------
# Flask route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    recommendations = []
    error = None

    if request.method == "POST":
        try:
            sleep_hours = float(request.form["sleep_hours"])
            screen_time = float(request.form["screen_time"])
            mood_score = int(request.form["mood_score"])
            activity_level = int(request.form["activity_level"])
            questionnaire_score = int(request.form["questionnaire_score"])

            features = [[
                sleep_hours,
                screen_time,
                mood_score,
                activity_level,
                questionnaire_score
            ]]

            prediction = model.predict(features)[0]
            recommendations = get_recommendation(prediction)

        except ValueError:
            error = "Please enter valid numeric values in all fields."

    return render_template_string(
        HTML_PAGE,
        prediction=prediction,
        recommendations=recommendations,
        error=error
    )

# -------------------------------
# Run app
# -------------------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)