import pandas as pd
import numpy as np

rows = 20000

timestamps = pd.date_range(
    start="2025-01-01",
    periods=rows,
    freq="h"
)

data = {
    "Timestamp": timestamps,

    "Travelator_ID": np.random.choice(
        ["T1", "T2", "T3"],
        rows
    ),

    "Passenger_Count": np.random.randint(
        10,
        300,
        rows
    ),

    "Occupancy_Percentage": np.random.randint(
        5,
        100,
        rows
    ),

    "Average_Travel_Time": np.random.randint(
        20,
        90,
        rows
    ),

    "Luggage_Count": np.random.randint(
        0,
        80,
        rows
    ),

    "Elderly_Count": np.random.randint(
        0,
        30,
        rows
    ),

    "Wheelchair_Count": np.random.randint(
        0,
        10,
        rows
    ),

    "Temperature": np.random.uniform(
        25,
        80,
        rows
    ),

    "Vibration": np.random.uniform(
        0.1,
        5.0,
        rows
    ),

    "Energy_Usage": np.random.uniform(
        10,
        100,
        rows
    ),

    "Is_Holiday": np.random.choice(
        [0,1],
        rows,
        p=[0.8,0.2]
    ),

    "Event_Level": np.random.choice(
        ["Low","Medium","High"],
        rows
    )
}

df = pd.DataFrame(data)

df["Day_Of_Week"] = df["Timestamp"].dt.day_name()

def congestion(row):

    if row["Occupancy_Percentage"] > 80:
        return "High"

    elif row["Occupancy_Percentage"] > 50:
        return "Medium"

    else:
        return "Low"

df["Congestion_Level"] = df.apply(
    congestion,
    axis=1
)

df.to_csv(
    "dataset/travelator_data.csv",
    index=False
)

print(df.head())
print("\nDataset Generated Successfully!")