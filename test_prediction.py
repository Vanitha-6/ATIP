import joblib
import pandas as pd

model = joblib.load("models/congestion_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

sample = pd.DataFrame([{
    "Passenger_Count": 250,
    "Occupancy_Percentage": 90,
    "Average_Travel_Time": 60,
    "Luggage_Count": 50,
    "Elderly_Count": 10,
    "Wheelchair_Count": 2,
    "Temperature": 45,
    "Vibration": 2.5,
    "Energy_Usage": 75,
    "Is_Holiday": 1
}])

prediction = model.predict(sample)

result = encoder.inverse_transform(prediction)

print("Predicted Congestion:", result[0])