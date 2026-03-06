import streamlit as st
import joblib
import numpy as np
import os

# Page config
st.set_page_config(page_title="Driver Risk Scoring", layout="centered")

# Load model
model_path = os.path.join("models", "random_forest.pkl")
model = joblib.load(model_path)

st.title("🚗 Driver Risk Scoring System")

st.write(
"""
This system predicts **driver risk level** using driving behaviour data
from **gyroscope and acceleration sensors**.
"""
)

st.divider()

# -----------------------------
# Sensor Input Section
# -----------------------------

st.header("Enter Driving Behaviour Data")

st.subheader("Gyroscope (Steering Movement)")

gyroX = st.slider("GyroX", -10.0, 10.0, 0.0)
gyroY = st.slider("GyroY", -10.0, 10.0, 0.0)
gyroZ = st.slider("GyroZ", -10.0, 10.0, 0.0)

st.subheader("Acceleration (Vehicle Motion)")

accX = st.slider("AccX", -20.0, 20.0, 0.0)
accY = st.slider("AccY", -20.0, 20.0, 0.0)
accZ = st.slider("AccZ", -20.0, 20.0, 0.0)

st.divider()

# -----------------------------
# Threshold Information
# -----------------------------

st.header("Driving Behaviour Guide")

st.info(
"""
**Safe Driving Behaviour**
- Smooth acceleration
- Small gyroscope movement
- Low sudden braking

**Risky Driving Behaviour**
- High acceleration spikes
- Aggressive steering
- Sudden braking or sharp turns
"""
)

st.markdown(
"""
Typical thresholds:

| Behaviour | Safe Range | Risky Range |
|-----------|-----------|-------------|
| Gyroscope | -3 to 3 | > 5 |
| Acceleration | -5 to 5 | > 10 |
"""
)

st.divider()

# -----------------------------
# Feature Engineering
# -----------------------------

def create_features(data):

    gyroX, gyroY, gyroZ, accX, accY, accZ = data

    acc_magnitude = np.sqrt(accX**2 + accY**2 + accZ**2)
    gyro_magnitude = np.sqrt(gyroX**2 + gyroY**2 + gyroZ**2)

    acc_intensity = abs(accX) + abs(accY) + abs(accZ)
    gyro_intensity = abs(gyroX) + abs(gyroY) + abs(gyroZ)

    movement_index = acc_magnitude * gyro_magnitude

    return np.array([
        gyroX, gyroY, gyroZ,
        accX, accY, accZ,
        acc_magnitude,
        gyro_magnitude,
        acc_intensity,
        gyro_intensity,
        movement_index
    ]).reshape(1, -1)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Driver Risk"):

    data = [gyroX, gyroY, gyroZ, accX, accY, accZ]

    features = create_features(data)

    prediction = model.predict(features)[0]

    risk_map = {
        1: "Safe Driver",
        2: "Low Risk Driver",
        3: "Moderate Risk Driver",
        4: "High Risk Driver"
    }

    risk_score = prediction * 25

    st.subheader("Prediction Result")

    st.success(f"Driver Type: {risk_map[prediction]}")

    st.metric("Risk Score", f"{risk_score}/100")

    # Risk message
    if prediction == 1:
        st.success("Driving behaviour is safe.")
    elif prediction == 2:
        st.info("Driver shows slightly risky behaviour.")
    elif prediction == 3:
        st.warning("Moderate risk driving detected.")
    else:
        st.error("High risk driving behaviour detected!")