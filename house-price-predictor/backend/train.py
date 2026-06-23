from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import pandas as pd

print("loading the dataset")
data = fetch_california_housing()

X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

print(f"size of X is {X.shape}")
print(f"size of y is {y.shape}")

#spliting
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)


#training the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train , y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test , y_pred)
r2 = r2_score(y_test , y_pred)

print(f"Average error: ${mae*100000:,.0f}")

joblib.dump(model , "house_model.joblib")
joblib.dump(list(X.columns) , "house_features.joblib")
