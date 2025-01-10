# analyze.py

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# ------------------------------------------------------------------
# 1) Confirm File Existence (Optional checks)
# ------------------------------------------------------------------
deadlines_path = "./Data/deadlines.json"
iphone_screen_time_path = "./Data/iPhoneScreenTime.json"
laptop_screen_time_path = "./Data/laptopScreenTime.json"
steps_kcal_path = "./Data/steps&kcal.json"

print("Checking file existence...")
print("deadlines.json:", os.path.exists(deadlines_path))
print("iPhoneScreenTime.json:", os.path.exists(iphone_screen_time_path))
print("laptopScreenTime.json:", os.path.exists(laptop_screen_time_path))
print("steps&kcal.json:", os.path.exists(steps_kcal_path))

# ------------------------------------------------------------------
# 2) Load the JSON Data
# ------------------------------------------------------------------
print("\nLoading data from JSON files...")

with open(deadlines_path, "r") as f:
    deadlines_data = json.load(f)  # { "2024-11-10": [...], "2024-11-22": [...], ... }

with open(iphone_screen_time_path, "r") as f:
    iphone_json = json.load(f)     # { "daily_records": [...], "metadata": {...} }

with open(laptop_screen_time_path, "r") as f:
    laptop_json = json.load(f)     # { "2024-11-04": "2 hours 34 minutes", ... }

with open(steps_kcal_path, "r") as f:
    steps_kcal_json = json.load(f) # { "2024-11-04": [2823, 94], ... }

# ------------------------------------------------------------------
# 3) Parse iPhone Screen Time Data
# ------------------------------------------------------------------
# The iPhone data has a "daily_records" list, each containing:
#   {
#     "date": "YYYY-MM-DD",
#     "day": "Monday",
#     "total_screen_time": "4:26:00",
#     "categories": {...},
#     "most_used_apps": [...]
#   }
#
# We'll flatten this into a DataFrame.
print("\nFlattening iPhone data into pandas DataFrame...")
iphone_records = iphone_json["daily_records"]
iphone_df = pd.json_normalize(iphone_records)

# For convenience, rename "date" to "Date" so that merges are consistent
iphone_df.rename(columns={"date": "Date"}, inplace=True)

# A helper function to convert "HH:MM:SS" to total minutes:
def hms_to_minutes(time_str):
    """Convert a string of the form 'H:M:S' to total minutes (float if needed)."""
    if not time_str or ":" not in time_str:
        return 0.0
    h, m, s = time_str.split(":")
    return int(h) * 60 + int(m) + float(s) / 60.0

iphone_df["total_screen_time_minutes"] = iphone_df["total_screen_time"].apply(hms_to_minutes)

print("iPhone Screen Time Data (flattened) sample:")
print(iphone_df[["Date", "day", "total_screen_time", "total_screen_time_minutes"]].head())

# ------------------------------------------------------------------
# 4) Parse Laptop Screen Time Data
# ------------------------------------------------------------------
# The laptop JSON has the form:
#   {
#     "2024-11-04": "2 hours 34 minutes",
#     "2024-11-05": "0 minutes",
#     ...
#   }
#
# We'll convert this to a DataFrame with columns: ["Date", "laptop_screen_time_minutes"].
print("\nParsing Laptop screen time data...")

laptop_records = []
for date_str, usage_str in laptop_json.items():
    # usage_str might be "X hours Y minutes" or "0 minutes", etc.
    # We'll convert to total minutes.
    # A quick function:
    def parse_laptop_time(t_str):
        if "hours" in t_str or "hour" in t_str:
            # e.g. "2 hours 34 minutes"
            parts = t_str.split()
            # possible patterns:
            #   ["2", "hours", "34", "minutes"]
            #   ["1", "hour",  "12", "minutes"]
            #   ["0", "minutes"]
            hours = 0
            minutes = 0
            if "hour" in parts[1]:
                hours = int(parts[0])
            if len(parts) >= 4 and "minute" in parts[3]:
                minutes = int(parts[2])
            return hours * 60 + minutes
        else:
            # e.g. "0 minutes" or "5 minutes"
            return int(t_str.split()[0])

    laptop_minutes = parse_laptop_time(usage_str)
    laptop_records.append({"Date": date_str, "laptop_screen_time_minutes": laptop_minutes})

laptop_df = pd.DataFrame(laptop_records)
print("Laptop Data sample:")
print(laptop_df.head())

# ------------------------------------------------------------------
# 5) Parse Steps & Calories Data
# ------------------------------------------------------------------
# The steps & kcal JSON has the form:
#   {
#     "2024-11-04": [2823, 94],
#     "2024-11-05": [9285, 365],
#     ...
#   }
# We'll convert to a DataFrame: columns => "Date", "steps", "kcal".
print("\nParsing Steps & Calories data...")

steps_kcal_records = []
for date_str, arr in steps_kcal_json.items():
    # arr might be [2823, 94]
    steps_val, kcal_val = arr[0], arr[1]
    steps_kcal_records.append({
        "Date": date_str,
        "steps": steps_val,
        "kcal_burned": kcal_val
    })

steps_kcal_df = pd.DataFrame(steps_kcal_records)
print("Steps & Calories Data sample:")
print(steps_kcal_df.head())

# ------------------------------------------------------------------
# 6) Parse Deadlines
# ------------------------------------------------------------------
# The deadlines JSON has the form:
#   {
#     "2024-11-10": ["Midterm Exam, DSA210", "Midterm Exam, CS307", ...],
#     "2024-11-22": ["PA2 Deadline, CS307"],
#     ...
#   }
# We might store it in a DataFrame with columns => "Date", "deadlines_list", "num_deadlines"
print("\nParsing deadlines data...")

deadlines_records = []
for date_str, events in deadlines_data.items():
    # events is a list of strings
    deadlines_records.append({
        "Date": date_str,
        "deadlines_list": events,
        "num_deadlines": len(events)
    })

deadlines_df = pd.DataFrame(deadlines_records)
print("Deadlines Data sample:")
print(deadlines_df.head())

# ------------------------------------------------------------------
# 7) Merge All Datasets
# ------------------------------------------------------------------
# We'll create one master DataFrame that merges:
#   iPhone DF, Laptop DF, StepsKcal DF, Deadlines DF
# on the "Date" field. We use outer merges or inner merges depending on your preference.
print("\nMerging all datasets into a single DataFrame...")

merged_df = (
    iphone_df
    .merge(laptop_df, on="Date", how="outer")
    .merge(steps_kcal_df, on="Date", how="outer")
    .merge(deadlines_df, on="Date", how="outer")
)

# Convert "Date" column to actual date type (optional, for easier time series manipulation)
merged_df["Date"] = pd.to_datetime(merged_df["Date"], format="%Y-%m-%d", errors="coerce")

# Sort by Date
merged_df = merged_df.sort_values(by="Date").reset_index(drop=True)

print("Merged DataFrame sample (columns):")
print(merged_df.columns)
print(merged_df.head())

# ------------------------------------------------------------------
# 8) Exploratory Data Analysis
# ------------------------------------------------------------------

# --- (A) Summary Statistics ---
print("\nSummary Statistics for numeric columns:")
print(merged_df.describe())

# --- (B) Correlation Heatmap ---
# We might focus on columns like "total_screen_time_minutes", "laptop_screen_time_minutes", "steps", "kcal_burned", etc.
print("\nCorrelation Heatmap among numeric columns...")

numeric_cols = ["total_screen_time_minutes", "laptop_screen_time_minutes", "steps", "kcal_burned", "num_deadlines"]
corr_matrix = merged_df[numeric_cols].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix: Screen Time vs. Steps/Kcal/Deadlines")
plt.show()

# --- (C) Time-Series Plot ---
print("\nPlotting Time Series to visualize daily changes...")

plt.figure(figsize=(12, 6))
plt.plot(merged_df["Date"], merged_df["total_screen_time_minutes"], label="iPhone Screen Time (min)")
plt.plot(merged_df["Date"], merged_df["laptop_screen_time_minutes"], label="Laptop Screen Time (min)")
plt.plot(merged_df["Date"], merged_df["steps"], label="Steps", color="green")
plt.scatter(merged_df["Date"], merged_df["num_deadlines"], label="# of Deadlines", color="red", marker="x")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Metric Value")
plt.title("Daily Metrics Over Time")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- (D) Basic Hypothesis Testing: "Does iPhone screen time correlate with steps?"
print("\nPearson Correlation: iPhone Screen Time vs. Steps")
valid_data = merged_df.dropna(subset=["total_screen_time_minutes", "steps"])
corr, pval = pearsonr(valid_data["total_screen_time_minutes"], valid_data["steps"])
print(f"Pearson correlation = {corr:.4f}, p-value = {pval:.4g}")
if pval < 0.05:
    print("-> Statistically significant correlation.")
else:
    print("-> Not statistically significant correlation (at alpha=0.05).")

# ------------------------------------------------------------------
# 9) Further Analysis / Exports
# ------------------------------------------------------------------
# You could export the merged DataFrame to CSV or Excel for deeper analysis:
# merged_df.to_csv("./Data/merged_analysis.csv", index=False)

print("\nAnalysis complete. You can now explore `merged_df` further for your research!")
