# Analytic Task: Defensive Performance – Prabodha

# Analytical Question:
# Do defenders and midfielders have a significant difference in interceptions per 90 minutes during FIFA World Cup 2026?

import pandas as pd
import os

# Define dataset path
# The dataset is stored inside the raw data folder

file_path = os.path.join(
    os.path.dirname(__file__),
    "../data/raw/world_cup_2026_defensive_actions_raw.csv"
)

# Read CSV file
df = pd.read_csv(file_path)

# Initial Dataset Exploration

# Display first five rows
print(df.head())

# Check dataset size

print(
    "Dataset size:",
    df.shape
)

# Display column names

print(
    df.columns.tolist()
)

# Display dataset information

df.info()

print(
    "Total Rows and Columns:",
    df.shape
)

# Select Required Variables
# Only variables required for this analysis are selected:

# Player  -> Player name
# Pos     -> Playing position
# Squad   -> Team name
# 90s     -> Playing time in 90-minute units
# Int     -> Number of interceptions

analysis_df = df[
    [
        "Player",
        "Pos",
        "Squad",
        "90s",
        "Int"
    ]
].copy()

# Check missing values

print(
    analysis_df.isnull().sum()
)

# Data Cleaning and Eligibility Criteria
# Convert numerical columns into numeric format
# Invalid values will be converted into missing values

analysis_df["90s"] = pd.to_numeric(
    analysis_df["90s"],
    errors="coerce"
)

analysis_df["Int"] = pd.to_numeric(
    analysis_df["Int"],
    errors="coerce"
)

# Eligibility criteria:

# Only players who:
# 1. Are defenders (DF) or midfielders (MF)
# 2. Played at least one full 90-minute match
# 3. Have valid interception data

# are included in the analysis.

eligible_df = analysis_df[
    (analysis_df["Pos"].isin(["DF", "MF"])) &
    (analysis_df["90s"] >= 1.0) &
    (analysis_df["Int"].notna())
].copy()

# Display final eligible sample distribution

print(
    eligible_df["Pos"].value_counts()
)