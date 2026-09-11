import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load model relative to current execution context
MODEL_PATH = os.path.join(os.path.dirname(__file__), "decision.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# HTML Layout and CSS embedded directly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #f4f7f6; color: #333; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
        .container { background: #ffffff; padding: 30px 40px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.1); max-width: 650px; width: 100%; }
        h2 { text-align: center; margin-bottom: 24px; color: #1a365d; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .form-group { display: flex; flex-direction: column; }
        .form-group.full-width { grid-column: span 2; }
        label { font-size: 0.9rem; font-weight: 600; margin-bottom: 6px; color: #4a5568; }
        input, select { padding: 10px 14px; border: 1px solid #cbd5e0; border-radius: 6px; font-size: 0.95rem; outline: none; transition: border-color 0.2s; }
        input:focus, select:focus { border-color: #3182ce; }
        button { grid-column: span 2; margin-top: 12px; padding: 12px; background-color: #3182ce; color: white; border: none; border-radius: 6px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background-color 0.2s; }
        button:hover { background-color: #2b6cb0; }
        .result-box { margin-top: 24px; padding: 16px; border-radius: 8px; text-align: center; font-size: 1.1rem; font-weight: bold; }
        .approved { background-color: #c6f6d5; color: #22543d; border: 1px solid #9ae6b4; }
        .rejected { background-color: #fed7d7; color: #742a2a; border: 1px solid #feb2b2; }
    </style>
</head>
<body>

<div class="container">
    <h2>Loan Approval Predictor</h2>
    <form method="POST" action="/predict">
        <div class="form-grid">
            <div class="form-group">
                <label for="no_of_dependents">Number of Dependents</label>
                <input type="number" name="no_of_dependents" id="no_of_dependents" required min="0" value="0">
            </div>

            <div class="form-group">
                <label for="education">Education (Categorical)</label>
                <select name="education" id="education" required>
                    <option value="1">Graduate</option>
                    <option value="0">Not Graduate</option>
                </select>
            </div>

            <div class="form-group">
                <label for="self_employed">Self Employed (Categorical)</label>
                <select name="self_employed" id="self_employed" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="income_annum">Annual Income ($)</label>
                <input type="number" name="income_annum" id="income_annum" required min="0">
            </div>

            <div class="form-group">
                <label for="loan_amount">Loan Amount ($)</label>
                <input type="number" name="loan_amount" id="loan_amount" required min="0">
            </div>

            <div class="form-group">
                <label for="cibil_score">CIBIL Score</label>
                <input type="number" name="cibil_score" id="cibil_score" required min="300" max="900">
            </div>

            <div class="form-group">
                <label for="residential_assets_value">Residential Asset Value ($)</label>
                <input type="number" name="residential_assets_value" id="residential_assets_value" required min="0">
            </div>

            <div class="form-group">
                <label for="commercial_assets_value">Commercial Asset Value ($)</label>
                <input type="number" name="commercial_assets_value" id="commercial_assets_value" required min="0">
            </div>

            <div class="form-group">
                <label for="luxury_assets_value">Luxury Asset Value ($)</label>
                <input type="number" name="luxury_assets_value" id="luxury_assets_value" required min="0">
            </div>

            <div class="form-group">
                <label for="bank_asset_value">Bank Asset Value ($)</label>
                <input type="number" name="bank_asset_value" id="bank_asset_value" required min="0">
            </div>

            <button type="submit">Predict Status</button>
        </div>
    </form>

    {% if prediction_text %}
        <div class="result-box {{ result_class }}">
            {{ prediction_text }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["no_of_dependents"]),
        float(request.form["education"]),
        float(request.form["self_employed"]),
        float(request.form["income_annum"]),
        float(request.form["loan_amount"]),
        float(request.form["cibil_score"]),
        float(request.form["residential_assets_value"]),
        float(request.form["commercial_assets_value"]),
        float(request.form["luxury_assets_value"]),
        float(request.form["bank_asset_value"]),
    ]

    final_features = [np.array(features)]
    prediction = model.predict(final_features)[0]

    # Map prediction binary value to user message
    if str(prediction).strip() in ["1", "1.0", "Approved"]:
        text = "Loan Status: Approved"
        css_class = "approved"
    else:
        text = "Loan Status: Rejected"
        css_class = "rejected"

    return render_template_string(
        HTML_TEMPLATE, prediction_text=text, result_class=css_class
    )

if __name__ == "__main__":
    app.run(debug=True)
