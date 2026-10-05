from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# =====================================================
# 1. CREATE SAMPLE SENSOR DATA
# =====================================================

np.random.seed(42)

n = 300

tilt_x = np.random.uniform(0, 5, n)
tilt_y = np.random.uniform(0, 5, n)
vibration = np.random.uniform(0, 10, n)
displacement = np.random.uniform(0, 10, n)
crack = np.random.uniform(0, 5, n)

risk = []

for i in range(n):

    score = (
        tilt_x[i]
        + tilt_y[i]
        + vibration[i]
        + displacement[i]
        + crack[i]
    )

    if score < 14:
        risk.append(0)

    elif score < 24:
        risk.append(1)

    else:
        risk.append(2)


df = pd.DataFrame({
    "tilt_x": tilt_x,
    "tilt_y": tilt_y,
    "vibration": vibration,
    "displacement": displacement,
    "crack": crack,
    "risk": risk
})


# =====================================================
# 2. FEATURES
# =====================================================

features = [
    "tilt_x",
    "tilt_y",
    "vibration",
    "displacement",
    "crack"
]

X = df[features]
Y = df["risk"]


# =====================================================
# 3. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)


# =====================================================
# 4. RANDOM FOREST MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)


# =====================================================
# 5. TRAIN MODEL
# =====================================================

model.fit(X_train, Y_train)


# =====================================================
# 6. MODEL ACCURACY
# =====================================================

Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)

print("--------------------------------")
print("MINE SUBSIDENCE AI SYSTEM")
print("--------------------------------")
print("Dataset samples:", n)
print("Model training completed")
print("Model accuracy:", round(accuracy * 100, 2), "%")


# =====================================================
# 7. SAVE MODEL
# =====================================================

joblib.dump(model, "mine_subsidence_model.pkl")

print("Model saved successfully")
print("--------------------------------")


# =====================================================
# 8. HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2)
    )


# =====================================================
# 9. PREDICTION API
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        tilt_x_value = float(data["tilt_x"])
        tilt_y_value = float(data["tilt_y"])
        vibration_value = float(data["vibration"])
        displacement_value = float(data["displacement"])
        crack_value = float(data["crack"])


        # Create DataFrame
        sample = pd.DataFrame(
            [[
                tilt_x_value,
                tilt_y_value,
                vibration_value,
                displacement_value,
                crack_value
            ]],
            columns=features
        )


        # AI prediction
        prediction = int(model.predict(sample)[0])


        # Probability
        probabilities = model.predict_proba(sample)[0]

        confidence = round(
            float(max(probabilities)) * 100,
            2
        )


        # Convert code to risk
        if prediction == 0:

            risk_name = "NORMAL"
            message = "Mine condition appears normal. Continue monitoring."

        elif prediction == 1:

            risk_name = "WARNING"
            message = "Warning detected. Increase monitoring and inspection."

        else:

            risk_name = "HIGH RISK"
            message = "High risk detected. Immediate engineering inspection is recommended."


        return jsonify({

            "prediction": risk_name,
            "risk_code": prediction,
            "confidence": confidence,
            "message": message

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# =====================================================
# 10. RUN SERVER
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )