from sklearn.linear_model import LinearRegression
import joblib
import os

# Create a very small dummy dataset
X = [[1], [2], [3], [4]]
y = [1, 2, 3, 4]

# Train a tiny model
model = LinearRegression()
model.fit(X, y)

# Ensure model_files folder exists
os.makedirs("model_files", exist_ok=True)

# Save the model
joblib.dump(model, "model_files/japan_eew_model.pkl")

print("Dummy model created successfully")
