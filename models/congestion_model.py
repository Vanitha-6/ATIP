import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset/travelator_data.csv")

# Features
X = df[
    [
        "Passenger_Count",
        "Occupancy_Percentage",
        "Average_Travel_Time",
        "Luggage_Count",
        "Elderly_Count",
        "Wheelchair_Count",
        "Temperature",
        "Vibration",
        "Energy_Usage",
        "Is_Holiday"
    ]
]

# Target
y = df["Congestion_Level"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(
    model,
    "models/congestion_model.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("Model Saved Successfully")