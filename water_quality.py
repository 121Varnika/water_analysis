import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Function to calculate WQI
def calculate_wqi(pH, TDS, Turbidity, Fe):
    pH_score = 100 - abs(7.5 - pH) * 20
    TDS_score = max(0, 100 - TDS / 5)
    Turbidity_score = max(0, 100 - Turbidity * 20)
    Fe_score = max(0, 100 - Fe * 333)
    return min(100, max(0, (pH_score * 0.3 + TDS_score * 0.2 + Turbidity_score * 0.3 + Fe_score * 0.2)))

# Save each entry to Excel
def save_to_excel(data_dict, filename='water_quality_log.xlsx'):
    df_new = pd.DataFrame([data_dict])
    if os.path.exists(filename):
        df_existing = pd.read_excel(filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_excel(filename, index=False)
    print(f"📁 Data saved to {filename}")

# Analysis function
def analyze_water_quality():
    global current_readings, wqi, status, current_month, monthly_avg

    # Generate dummy historical data
    historical_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
        'pH': np.random.uniform(6.5, 8.5, 365),
        'TDS': np.random.uniform(100, 500, 365),
        'Turbidity': np.random.uniform(0.1, 5.0, 365),
        'Fe': np.random.uniform(0.01, 0.3, 365)
    })
    historical_data['Month'] = historical_data['Date'].dt.month_name()
    monthly_avg = historical_data.groupby('Month').mean(numeric_only=True).reset_index()
    current_month = datetime.now().strftime('%B')

    # User input
    print("\nEnter current water quality readings:")
    current_readings = {
        'pH': float(input("pH (6.5-8.5): ")),
        'TDS': float(input("TDS (100-500 ppm): ")),
        'Turbidity': float(input("Turbidity (0.1-5.0 NTU): ")),
        'Fe': float(input("Iron content (0.01-0.3 mg/L): "))
    }

    # WQI Calculation
    wqi = calculate_wqi(**current_readings)
    status = "Excellent" if wqi >= 90 else "Good" if wqi >= 70 else "Fair" if wqi >= 50 else "Poor"

    # Display Results
    print("\n" + "=" * 50)
    print(f"WQI: {wqi:.2f} - Status: {status}")
    print("=" * 50)

    # Alerts and Recommendations
    alerts = []
    recommendations = []

    if current_readings['pH'] < 6.5:
        alerts.append("🧪 pH is acidic — may increase iron solubility")
        recommendations.append("Add pH increaser (soda ash)")
    elif current_readings['pH'] > 8.5:
        alerts.append("🧪 pH is alkaline — may cause scaling")
        recommendations.append("Add pH decreaser (muriatic acid)")

    if current_readings['TDS'] > 500:
        alerts.append("📈 TDS above safe limit - water may taste salty")
        recommendations.append("Install RO system or distiller")

    if current_readings['Turbidity'] > 5:
        alerts.append("🌫 High turbidity - possible contamination")
        recommendations.append("Improve filtration (sand/activated carbon)")

    if current_readings['Fe'] > 0.3:
        alerts.append("⚠️ Iron (Fe) is very high - this can stain pipes")
        recommendations.append("Use iron filter or oxidize (aeration + filtration)")

    if alerts:
        print("\n🔍 ALERTS:")
        for alert in alerts:
            print(alert)

    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"- {rec}")

    # Plotting
    features = ['pH', 'TDS', 'Turbidity', 'Fe']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#3498db', '#2ecc71']
    threshold_color = '#e74c3c'

    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        current_val = current_readings[feature]
        avg_val = monthly_avg[monthly_avg['Month'] == current_month][feature].values[0]
        bars = ax.bar([0, 1], [current_val, avg_val],
                      width=0.5,
                      color=colors,
                      edgecolor='white',
                      linewidth=2,
                      zorder=3)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#dddddd')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['CURRENT', 'AVG'],
                           fontsize=10,
                           fontweight='bold',
                           rotation=45)
        ax.set_title(feature.upper(),
                     fontsize=12,
                     fontweight='bold',
                     pad=12,
                     color='#333333')
        ax.set_ylabel('VALUE',
                      fontsize=9,
                      labelpad=8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + 0.05 * max(current_val, avg_val),
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    color='#333333')
        if feature == 'pH':
            ax.axhspan(6.5, 8.5, facecolor='#2ecc7055', zorder=1)
            ax.axhline(6.5, color=threshold_color, linestyle=':', linewidth=1.5, zorder=2)
            ax.axhline(8.5, color=threshold_color, linestyle=':', linewidth=1.5, zorder=2)
            ax.text(0.5, 6.3, 'SAFE RANGE',
                    ha='center',
                    fontsize=8,
                    color=threshold_color)
        else:
            threshold = 500 if feature == 'TDS' else 5.0 if feature == 'Turbidity' else 0.3
            ax.axhline(threshold,
                       color=threshold_color,
                       linestyle=':',
                       linewidth=1.5,
                       zorder=2)
            ax.text(1.1, threshold * 1.05, 'MAX THRESHOLD',
                    ha='right',
                    fontsize=8,
                    color=threshold_color)

    fig.suptitle(f'WATER QUALITY ANALYSIS | {current_month.upper()} {datetime.now().year}\nWQI: {wqi:.1f} ({status})',
                 y=0.98,
                 fontsize=14,
                 fontweight='bold',
                 color='#333333')

    plt.tight_layout(pad=3)
    plt.show()

# Run the input loop
def run_interactive_session():
    print("💧 SMART WATER QUALITY MONITORING SYSTEM 💧")
    while True:
        analyze_water_quality()

        # Add metadata and save
        current_readings['Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_readings['WQI'] = wqi
        current_readings['Status'] = status
        save_to_excel(current_readings)

        # Continue prompt
        repeat = input("\nDo you want to enter another reading? (y/n): ").strip().lower()
        if repeat != 'y':
            print("🚪 Exiting the session. Bye!")
            break

# Start the program
run_interactive_session()

