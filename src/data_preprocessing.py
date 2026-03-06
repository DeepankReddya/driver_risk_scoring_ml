import pandas as pd


def load_data(path):
    """Load dataset"""
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """Clean dataset and separate features and labels"""

    # Rename target column for easier access
    df = df.rename(columns={"Target(Class)": "target"})

    # Remove missing values
    df = df.dropna()

    # Split features and labels
    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


if __name__ == "__main__":

    df = load_data("data/driving_data.csv")

    X, y = preprocess_data(df)

    print("Dataset shape:", df.shape)
    print("Feature columns:", X.columns)