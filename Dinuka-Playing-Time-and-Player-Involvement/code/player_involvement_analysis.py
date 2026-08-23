"""
FIFA World Cup 2026 - Analytic Task: Playing Time & Player Involvement
Author: Dinuka

Analytic question:
    Is there a significant difference in the average minutes played per
    appearance (Mn/MP) between Defenders (DF) and Midfielders (MF)
    during the FIFA World Cup 2026?

Data source:
    FBref World Cup 2026 Playing Time table
"""

# Import required libraries
import pandas as pd

##### DATA LOADING #####

# Reproducibility and statistical settings
RANDOM_SEED = 42
SAMPLE_SIZE_PER_GROUP = 150
ALPHA = 0.05

# 95% confidence level when alpha = 0.05
CONFIDENCE_LEVEL = 1 - ALPHA

# Raw dataset path
RAW_PATH = "../data/raw/world_cup_2026_player_involvement_raw.csv"

# Load dataset
raw = pd.read_csv(RAW_PATH)

print("=" * 70)
print("RAW DATA LOADED")
print("=" * 70)

print(f"Number of rows: {raw.shape[0]}")
print(f"Number of columns: {raw.shape[1]}")

print("\nFirst 5 rows:")
print(raw.head())

print("\nColumn names:")
print(raw.columns.tolist())

print("\nDataset information:")
raw.info()

print("\nMissing values in raw dataset:")
print(raw.isnull().sum())

##### DATA WRANGLING AND CLEANING #####

print("\n" + "=" * 70)
print("DATA WRANGLING AND CLEANING")
print("=" * 70)


# Select variables related to the analytical question
columns_needed = [
    "Player",
    "Pos",
    "MP",
    "Mn/MP",
    "Player_ID"
]

df = raw[columns_needed].copy()

# Clean position values
df["Pos"] = (
    df["Pos"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# Check duplicate rows
duplicate_rows = df.duplicated().sum()

print(
    f"\nExact duplicate rows found: "
    f"{duplicate_rows}"
)

# Remove exact duplicate rows if any exist
df = df.drop_duplicates().copy()