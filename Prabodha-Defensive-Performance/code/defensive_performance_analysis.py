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

