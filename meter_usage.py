import matplotlib.pyplot as plt
from datetime import datetime

# In-memory storage grouped by year
all_data = {}

# Valid month strings
valid_months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# 📊 Function to plot meter usage
def plot_utilization(data):
    if not data:
        print("🚫 No data available for this year.")
        return

    meter_keys = [f"M{i}" for i in range(1, 6)]
    cumulative = {key: 0 for key in meter_keys}
    total_usage = 0

    for entry in data.values():  # dict of month -> readings
        for key in meter_keys:
            cumulative[key] += entry[key]
            total_usage += entry[key]

    if total_usage == 0:
        print("🚿 All readings are zero!")
        return

    utilization = [cumulative[key] / total_usage for key in meter_keys]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(meter_keys, utilization, color="steelblue")

    bars[utilization.index(max(utilization))].set_color("green")
    bars[utilization.index(min(utilization))].set_color("red")

    avg = 1 / len(meter_keys)
    plt.axhline(avg, color="red", linestyle="--", label="Average Utilization")

    for i, val in enumerate(utilization):
        plt.text(i, val + 0.01, f"{val:.2%}", ha="center")

    plt.title("💧 Water Meter Utilization (This Year)")
    plt.xlabel("Meters")
    plt.ylabel("Utilization %")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.close()

# 🔁 Input Loop
current_year = datetime.now().year
while True:
    print(f"\n📆 Current Year: {current_year}")
    print("📝 Enter water readings for a month:")

    month = input("Month (or type 'exit' to quit): ").strip()

    if month.lower() == "exit":
        print("👋 Bye! See you next time.")
        break

    # Validate month
    if not isinstance(month, str) or month.capitalize() not in [m.capitalize() for m in valid_months]:
        print("❌ Invalid month! Please enter a valid month name like 'Jan', 'February', etc.")
        continue

    # Normalize month name (e.g., 'jan' → 'January')
    month = month.capitalize()
    if month in ["Jan", "January"]:
        month = "January"
    elif month in ["Feb", "February"]:
        month = "February"
    elif month in ["Mar", "March"]:
        month = "March"
    elif month in ["Apr", "April"]:
        month = "April"
    elif month == "May":
        month = "May"
    elif month in ["Jun", "June"]:
        month = "June"
    elif month in ["Jul", "July"]:
        month = "July"
    elif month in ["Aug", "August"]:
        month = "August"
    elif month in ["Sep", "September"]:
        month = "September"
    elif month in ["Oct", "October"]:
        month = "October"
    elif month in ["Nov", "November"]:
        month = "November"
    elif month in ["Dec", "December"]:
        month = "December"

    # Initialize year if not already
    if current_year not in all_data:
        all_data[current_year] = {}

    # Prevent duplicate month entry
    if month in all_data[current_year]:
        print(f"⚠️ Month '{month}' already entered for {current_year}. Try another month.")
        continue

    # Accept meter readings
    readings = {}
    for i in range(1, 6):
        val = float(input(f"Meter M{i} reading: "))
        readings[f"M{i}"] = val

    all_data[current_year][month] = readings

    # 🖼️ Show updated analysis
    print("📊 Generating cumulative usage chart...")
    plot_utilization(all_data[current_year])

    # Check if December has been entered
    if "December" in all_data[current_year]:
        current_year += 1
