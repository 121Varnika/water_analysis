import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize an empty DataFrame to store the data
columns = ['pH', 'TDS', 'Turbidity', 'Fe', 'WQI', 'Status']
data = pd.DataFrame(columns=columns)

# Function to calculate WQI
def calculate_wqi(pH, TDS, turbidity, Fe):
    weights = {
        'pH': 0.2,
        'TDS': 0.3,
        'Turbidity': 0.3,
        'Fe': 0.2
    }

    # Normalize and clip to avoid weird extremes
    normalized_ph = np.clip(((pH - 6.5) / (8.5 - 6.5)) * 100, 0, 100)
    normalized_tds = np.clip(((TDS - 0) / (500 - 0)) * 100, 0, 100)
    normalized_turbidity = np.clip(((turbidity - 0) / (5 - 0)) * 100, 0, 100)
    normalized_fe = np.clip(((Fe - 0) / (0.3 - 0)) * 100, 0, 100)

    # Weighted scores
    WQI = (normalized_ph * weights['pH'] +
           normalized_tds * weights['TDS'] +
           normalized_turbidity * weights['Turbidity'] +
           normalized_fe * weights['Fe'])
    return WQI

# WQI status checker
def interpret_wqi(WQI):
    if WQI <= 50:
        return "Excellent 💎"
    elif WQI <= 100:
        return "Good ✅"
    elif WQI <= 200:
        return "Poor ⚠️"
    elif WQI <= 300:
        return "Very Poor 🚫"
    else:
        return "Unfit for Consumption ☠️"

# Spike check
def check_for_spike(new_value, previous_value, threshold=20):
    return abs(new_value - previous_value) > threshold

# === USER INPUT LOOP ===
while True:
    # Get input
    pH = float(input("Enter pH value: "))
    TDS = float(input("Enter TDS value (mg/L): "))
    turbidity = float(input("Enter Turbidity value (NTU): "))
    Fe = float(input("Enter Fe value (mg/L): "))

    # Calculate WQI and status
    WQI = calculate_wqi(pH, TDS, turbidity, Fe)
    status = interpret_wqi(WQI)

    # Add to DataFrame
    new_entry = pd.DataFrame([[pH, TDS, turbidity, Fe, WQI, status]], columns=columns)
    data = pd.concat([data, new_entry], ignore_index=True)

    # === Smart Alerts ===
    print(f"\n💧 WQI: {WQI:.2f} — Status: {status}")

    if len(data) > 1:
        if check_for_spike(TDS, data['TDS'].iloc[-2]):
            print("🚨 Alert: TDS spike detected!")
        if check_for_spike(turbidity, data['Turbidity'].iloc[-2]):
            print("🚨 Alert: Turbidity spike detected!")

    # 🌡 Inter-feature awareness
    if pH < 6.5:
        print("🧪 pH is acidic — this might cause iron (Fe) to increase.")
    if Fe > 0.3:
        print("⚠️ Iron (Fe) is very high — this can stain pipes and affect taste.")
    if TDS > 500:
        print("📈 TDS above safe limit — water may taste salty or bitter.")
    if turbidity > 5:
        print("🌫 High turbidity — possible contamination or poor filtration.")

    # Plot scatter correlations
    if len(data) > 1:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(data['TDS'], data['Turbidity'], color='blue')
        plt.title('TDS vs Turbidity')
        plt.xlabel('TDS (mg/L)')
        plt.ylabel('Turbidity (NTU)')

        plt.subplot(1, 2, 2)
        plt.scatter(data['pH'], data['Fe'], color='green')
        plt.title('pH vs Fe')
        plt.xlabel('pH')
        plt.ylabel('Fe (mg/L)')

        plt.tight_layout()
        plt.show()

        # 💥 Correlation heatmap
        print("\n🔍 Correlation between features:")
        plt.figure(figsize=(6, 4))
        sns.heatmap(data[columns[:-2]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    # Continue or not
    cont = input("\n➡️  Enter another set? (yes/no): ")
    if cont.lower() != 'yes':
        break

# Save data
data.to_csv("water_quality_data.csv", index=False)
print("\n✅ Data saved to 'water_quality_data.csv'")
