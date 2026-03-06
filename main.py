from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_features
from src.train_model import train_models
from src.evaluate_model import evaluate


def main():

    # Load data
    df = load_data("data/driving_data.csv")

    # Preprocess
    X, y = preprocess_data(df)

    # Feature engineering
    X = create_features(X)

    # Train models
    models, X_test, y_test = train_models(X, y)

    # Evaluate
    evaluate(models, X_test, y_test)


if __name__ == "__main__":
    main()