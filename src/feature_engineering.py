import numpy as np


def create_features(X):

    # Acceleration magnitude
    X["acc_magnitude"] = np.sqrt(
        X["AccX"]**2 + X["AccY"]**2 + X["AccZ"]**2
    )

    # Gyroscope magnitude
    X["gyro_magnitude"] = np.sqrt(
        X["GyroX"]**2 + X["GyroY"]**2 + X["GyroZ"]**2
    )

    # Acceleration intensity
    X["acc_intensity"] = (
        abs(X["AccX"]) + abs(X["AccY"]) + abs(X["AccZ"])
    )

    # Gyro intensity
    X["gyro_intensity"] = (
        abs(X["GyroX"]) + abs(X["GyroY"]) + abs(X["GyroZ"])
    )

    # Combined movement feature
    X["movement_index"] = X["acc_magnitude"] * X["gyro_magnitude"]

    return X