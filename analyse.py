# Import necessary libraries
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
print(os.path.exists("./Data/deadlines.json"))  # Should return True
print(os.path.exists("./Data/iPhoneScreenTime.json"))  # Should return True

# File paths
deadlines_path = "./Data/deadlines.json"
iphone_screen_time_path = "./Data/iPhoneScreenTime.json"
laptop_screen_time_path = "./Data/laptopScreenTime.json"
steps_kcal_path = "./Data/steps&kcal.json"

# Load JSON data
print("Loading data...")
with open(deadlines_path, "r") as file:
    deadlines = json.load(file)

iphone_data = pd.read_json(iphone_screen_time_path)
laptop_data = pd.read_json(laptop_screen_time_path)
steps_kcal_data = pd.read_json(steps_kcal_path)

print("iPhone Data Sample:")
print(iphone_data.head())

print("\nLaptop Data Sample:")
print(laptop_data.head())

print("\nSteps & Calories Data Sample:")
print(steps_kcal_data.head())

# Flatten iPhone data
print("\nFlattening iPhone data...")
iphone_flat = pd.json_normalize(iphone_data["daily_records"])
print("Flattened iPhone Data Sample:")
print(iphone_flat.head())

# Convert time strings to minutes
print("\nConverting time strings to minutes...")
def time_to_minutes(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 60 + m + s / 60

iphone_flat["total_screen_time_minutes"] = iphone_flat["total_screen_time"].apply(time_to_minutes)
print("iPhone Data with Total Screen Time in Minutes:")
print(iphone_flat[["date", "total_screen_time_minutes"]].head())

# Separate steps and calories
print("\nProcessing Steps and Calories data...")
def extract_steps_kcal(value):
    steps, kcal = value.split(",")
    steps = int(steps.split("(")[0].strip())
    kcal = int(kcal.split("(")[0].strip())
    return steps, kcal

steps_kcal_data[["steps", "kcal_burned"]] = steps_kcal_data["value"].apply(
    lambda x: pd.Series(extract_steps_kcal(x))
)
print("Processed Steps & Calories Data:")
print(steps_kcal_data[["Date", "steps", "kcal_burned"]].head())

# Merge all datasets
print("\nMerging datasets...")
iphone_flat = iphone_flat.rename(columns={"date": "Date"})
merged_data = (
    iphone_flat.merge(steps_kcal_data, on="Date", how="inner")
    .merge(laptop_data, on="Date", how="inner")
)
print("Merged Data Sample:")
print(merged_data.head())

# Exploratory Data Analysis
## Summary Statistics
print("\nSummary Statistics:")
print(merged_data.describe())

## Time Series Analysis
print("\nPlotting Time Series Analysis...")
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
print("\nGenerating Correlation Heatmap...")
correlation_matrix = merged_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Hypothesis Testing
print("\nPerforming Hypothesis Testing...")
correlation, p_value = pearsonr(
    merged_data["total_screen_time_minutes"], merged_data["steps"]
)
print(f"Correlation between iPhone Screen T
