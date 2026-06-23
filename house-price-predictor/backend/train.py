from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd

def train_and_save_model():
    print("Training model...")

    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "house_model.joblib")
    joblib.dump(list(X.columns), "house_features.joblib")

    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
