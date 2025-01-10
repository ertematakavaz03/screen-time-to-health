# Import necessary libraries
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# File paths
deadlines_path = "./Data/deadlines.json"
iphone_screen_time_path = "./Data/iPhoneScreenTime.json"
laptop_screen_time_path = "./Data/laptopScreenTime.json"
steps_kcal_path = "./Data/steps&kcal.json"

# Load JSON data
with open(deadlines_path, "r") as file:
    deadlines = json.load(file)

iphone_data = pd.read_json(iphone_screen_time_path)
laptop_data = pd.read_json(laptop_screen_time_path)
steps_kcal_data = pd.read_json(steps_kcal_path)

# Flatten iPhone data
iphone_flat = pd.json_normalize(iphone_data["daily_records"])

# Convert time strings to minutes
def time_to_minutes(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 60 + m + s / 60

iphone_flat["total_screen_time_minutes"] = iphone_flat["total_screen_time"].apply(time_to_minutes)

# Separate steps and calories
def extract_steps_kcal(value):
    steps, kcal = value.split(",")
    steps = int(steps.split("(")[0].strip())
    kcal = int(kcal.split("(")[0].strip())
    return steps, kcal

steps_kcal_data[["steps", "kcal_burned"]] = steps_kcal_data["value"].apply(
    lambda x: pd.Series(extract_steps_kcal(x))
)

# Merge all datasets
iphone_flat = iphone_flat.rename(columns={"date": "Date"})
steps_kcal_data = steps_kcal_data.rename(columns={"Date": "Date"})
merged_data = (
    iphone_flat.merge(steps_kcal_data, on="Date", how="inner")
    .merge(laptop_data, on="Date", how="inner")
)

# Exploratory Data Analysis
## Summary Statistics
print("Summary Statistics:")
print(merged_data.describe())

## Time Series Analysis
plt.figure(figsize=(12, 6))
plt.plot(merged_data["Date"], merged_data["total_screen_time_minutes"], label="iPhone Screen Time")
plt.plot(merged_data["Date"], merged_data["screen_time_minutes"], label="Laptop Screen Time")
plt.plot(merged_data["Date"], merged_data["steps"], label="Steps")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Metrics")
plt.title("Time Series Analysis")
plt.grid()
plt.show()

## Correlation Heatmap
correlation_matrix = merged_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Hypothesis Testing
## Pearson Correlation Between Screen Time and Steps
correlation, p_value = pearsonr(
    merged_data["total_screen_time_minutes"], merged_data["steps"]
)
print(f"Correlation: {correlation}, P-value: {p_value}")

# Weekday vs Weekend Analysis
merged_data["Date"] = pd.to_datetime(merged_data["Date"])
merged_data["Weekday"] = merged_data["Date"].dt.weekday
merged_data["Weekend"] = merged_data["Weekday"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")

# Group by Weekday and Weekend
grouped = merged_data.groupby("Weekend").mean()
plt.figure(figsize=(10, 6))
grouped[["total_screen_time_minutes", "screen_time_minutes", "steps"]].plot(kind="bar")
plt.title("Weekday vs Weekend Analysis")
plt.xlabel("Day Type")
plt.ylabel("Average Metrics")
plt.grid()
plt.show()

# Save Processed Data
merged_data.to_csv("processed_data.csv", index=False)
