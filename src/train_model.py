import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def train_models(X, y):
    """
    Train multiple machine learning models and save them.
    """

    # Ensure models directory exists (important for GitHub Actions)
    os.makedirs("models", exist_ok=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Define models
    models = {
        "logistic": LogisticRegression(max_iter=1000),

        "decision_tree": DecisionTreeClassifier(
            random_state=42
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42
        ),

        "svm": SVC(
            kernel="rbf",
            probability=True
        )
    }

    trained_models = {}

    # Train each model
    for name, model in models.items():

        print(f"\nTraining {name} model...")

        model.fit(X_train, y_train)

        trained_models[name] = model

        # Save model
        model_path = f"models/{name}.pkl"
        joblib.dump(model, model_path)

        print(f"{name} model saved to {model_path}")

    return trained_models, X_test, y_test