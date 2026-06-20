
import pandas as pd
import joblib
import random
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
model = joblib.load("models/congestion_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.form

    # Input values
    passenger_count = int(data["Passenger_Count"])
    occupancy = int(data["Occupancy_Percentage"])
    travel_time = int(data["Average_Travel_Time"])
    luggage = int(data["Luggage_Count"])
    elderly = int(data["Elderly_Count"])
    wheelchair = int(data["Wheelchair_Count"])
    temperature = float(data["Temperature"])
    vibration = float(data["Vibration"])
    energy_usage = float(data["Energy_Usage"])
    is_holiday = int(data["Is_Holiday"])

    # Model Input
    sample = pd.DataFrame([{
        "Passenger_Count": passenger_count,
        "Occupancy_Percentage": occupancy,
        "Average_Travel_Time": travel_time,
        "Luggage_Count": luggage,
        "Elderly_Count": elderly,
        "Wheelchair_Count": wheelchair,
        "Temperature": temperature,
        "Vibration": vibration,
        "Energy_Usage": energy_usage,
        "Is_Holiday": is_holiday
    }])

    # Congestion Prediction
    prediction = model.predict(sample)
    congestion_level = encoder.inverse_transform(prediction)[0]

    # Future Prediction
    future_15 = random.randint(75, 95)
    future_30 = random.randint(60, 90)

    expected_passengers = passenger_count + random.randint(20, 100)

    growth_rate = round(
        ((expected_passengers - passenger_count)
         / passenger_count) * 100, 2
    )

    if growth_rate > 50:
        risk_level = "High"
    elif growth_rate > 25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Passenger Comfort Index
    comfort_score = 100
    comfort_score -= occupancy * 0.4
    comfort_score -= luggage * 0.2
    comfort_score = max(0, round(comfort_score, 2))

    if comfort_score >= 80:
        comfort_status = "Excellent"
    elif comfort_score >= 60:
        comfort_status = "Good"
    elif comfort_score >= 40:
        comfort_status = "Moderate"
    else:
        comfort_status = "Poor"

    # Accessibility Intelligence
    accessibility_score = 100

    accessibility_score -= elderly * 0.2
    accessibility_score -= wheelchair * 0.8

    accessibility_score = max(
        0,
        min(100, round(accessibility_score, 2))
    )

    if accessibility_score >= 80:
        accessibility_status = "Good"
    elif accessibility_score >= 60:
        accessibility_status = "Moderate"
    else:
        accessibility_status = "Needs Attention"

    # Health Intelligence
    temp_penalty = max(0, temperature - 40) * 1.2
    vibration_penalty = vibration * 8

    health_score = 100 - temp_penalty - vibration_penalty

    health_score = max(10, min(100, round(health_score, 2)))

    if health_score >= 80:
        health_status = "Healthy"
    elif health_score >= 60:
        health_status = "Monitor"
    else:
        health_status = "Maintenance Required"

    # Congestion Color
    if congestion_level == "High":
        congestion_color = "high"
    elif congestion_level == "Medium":
        congestion_color = "medium"
    else:
        congestion_color = "low"

    # =====================================
    # SMART AI DECISION ENGINE
    # =====================================

    all_suggestions = []
    priority = "Low"
    overall_status = "Normal Operations"

    # Critical Congestion
    if congestion_level == "High":

        priority = "Critical"
        overall_status = "Crowd Management Required"

        all_suggestions.extend([
            "Open alternate passenger routes immediately",
            "Activate digital signage for passenger redirection",
            "Deploy operational staff near travelator entry points",
            "Increase monitoring frequency",
            "Enable congestion alert notifications"
        ])

    elif congestion_level == "Medium":

        priority = "Medium"

        all_suggestions.extend([
            "Monitor passenger flow continuously",
            "Prepare alternate routing plans",
            "Optimize passenger guidance"
        ])

    # Passenger Comfort
    if comfort_score < 60:

        all_suggestions.extend([
            "Reduce passenger density",
            "Improve passenger spacing",
            "Optimize travelator utilization"
        ])

    # Accessibility
    if accessibility_score < 60:

        all_suggestions.extend([
            "Provide wheelchair assistance",
            "Allocate accessibility support staff",
            "Enable visual and audio guidance",
            "Improve boarding support"
        ])

    # Travelator Health
    if health_score < 70:

        if priority != "Critical":
            priority = "High"

        all_suggestions.extend([
            "Schedule preventive maintenance",
            "Inspect vibration levels",
            "Check motor temperature",
            "Perform component diagnostics",
            "Monitor energy consumption"
        ])

    # Holiday Traffic
    if is_holiday == 1:

        all_suggestions.extend([
            "Increase operational readiness",
            "Enable peak-hour monitoring mode",
            "Prepare additional staff support"
        ])

    # Future Forecast
    if future_30 > 85:

        all_suggestions.extend([
            "Prepare overflow passenger management plan",
            "Activate predictive congestion alerts",
            "Monitor passenger growth trend"
        ])

    # Remove duplicates
    all_suggestions = list(dict.fromkeys(all_suggestions))

    # =====================================
    # SELECT ONLY TOP 5 RECOMMENDATIONS
    # =====================================

    if priority == "Critical":

        suggestions = [
            "Open alternate passenger routes immediately",
            "Activate digital signage for passenger redirection",
            "Deploy operational staff near travelator entry points",
            "Enable congestion alert notifications",
            "Prepare overflow passenger management plan"
        ]

    elif priority == "High":

        suggestions = [
            "Schedule preventive maintenance",
            "Inspect vibration levels",
            "Check motor temperature",
            "Perform component diagnostics",
            "Monitor energy consumption"
        ]

    elif priority == "Medium":

        suggestions = [
            "Monitor passenger flow continuously",
            "Prepare alternate routing plans",
            "Optimize passenger guidance",
            "Increase operational readiness",
            "Enable peak-hour monitoring mode"
        ]

    else:

        suggestions = [
            "Travelator operating normally",
            "Continue routine monitoring",
            "Maintain current passenger flow",
            "Track travelator health metrics",
            "Review analytics dashboard periodically"
        ]

    # Executive Recommendation
    if priority == "Critical":

        recommendation = (
            "Immediate operational intervention required. "
            "High congestion risk detected."
        )

    elif priority == "High":

        recommendation = (
            "Maintenance and operational review recommended."
        )

    elif priority == "Medium":

        recommendation = (
            "Moderate operational adjustments recommended."
        )

    else:

        recommendation = (
            "Travelator operating within normal parameters."
        )

    return render_template(
        "result.html",

        congestion_level=congestion_level,
        congestion_color=congestion_color,

        comfort_score=comfort_score,
        comfort_status=comfort_status,

        accessibility_score=accessibility_score,
        accessibility_status=accessibility_status,

        health_score=health_score,
        health_status=health_status,

        recommendation=recommendation,

        future_15=future_15,
        future_30=future_30,
        expected_passengers=expected_passengers,

        growth_rate=growth_rate,
        risk_level=risk_level,
        current_passengers=passenger_count,

        priority=priority,
        overall_status=overall_status,
        suggestions=suggestions
    )


if __name__ == "__main__":
    app.run(debug=True)
