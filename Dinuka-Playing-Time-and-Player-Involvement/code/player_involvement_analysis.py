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

#Convert numerical variables to numeric data types
numeric_columns = ["MP", "Mn/MP"]
for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Check duplicate rows
duplicate_rows = df.duplicated().sum()
print(
    f"\nExact duplicate rows found: "
    f"{duplicate_rows}"
)

# Remove exact duplicate rows if any exist
df = df.drop_duplicates().copy()


# Check Player_ID values
missing_player_ids = df["Player_ID"].isnull().sum()
duplicate_player_ids = (
    df["Player_ID"]
    .dropna()
    .duplicated()
    .sum()
)
print(
    f"Missing Player_ID values: "
    f"{missing_player_ids}"
)
print(
    f"Duplicate Player_ID values: "
    f"{duplicate_player_ids}"
)

# Missing value check before filtering
print("\nMissing values before filtering:")
print(df.isnull().sum())


# Removing players who did not appear in a match
before = len(df)
df = df[
    df["MP"] > 0
].copy()
print(
    f"\nPlayers with zero appearances removed: "
    f"{before - len(df)}"
)


# Removing missing or invalid Mn/MP values
before = len(df)
df = df.dropna(
    subset=["Mn/MP", "Pos"]
).copy()
df = df[
    df["Mn/MP"] > 0
].copy()
print(
    f"Records with missing or invalid Mn/MP removed: "
    f"{before - len(df)}"
)


# Keep Defenders and Midfielders only - Only players classified exactly as DF or MF are included.
#
# Mixed-position players such as DFMF and MFDF are excluded because
# they cannot be clearly assigned to one comparison group.
before = len(df)
df = df[
    df["Pos"].isin(["DF", "MF"])
].copy()
print(
    f"Mixed-position and other-position records removed: "
    f"{before - len(df)}"
)

# Check the cleaned dataset
print(
    f"\nCleaned dataset size: "
    f"{df.shape[0]} rows"
)
print("\nEligible players by position:")
print(
    df["Pos"].value_counts()
)
print("\nMn/MP summary:")
print(
    df["Mn/MP"].describe()
)

# Save cleaned dataset
df.to_csv(
    "../data/processed/world_cup_2026_playing_time_clean.csv",
    index=False
)
print(
    "\nCleaned dataset saved as: "
    "../data/processed/world_cup_2026_playing_time_clean.csv"
)

##### POPULATION AND SAMPLING #####
print("=" * 70)
print("POPULATION AND SAMPLING")
print("=" * 70)


# Define the eligible population
#
# 1. Players classified exactly as DF or MF
# 2. Players who appeared in at least one match
# 3. Players with a valid positive Mn/MP value
population = df.copy()
population_summary = (
    population
    .groupby("Pos")["Mn/MP"]
    .agg(
        Count="count",
        Mean="mean",
        Standard_Deviation="std"
    )
)
print("\nEligible population:")
print(population_summary.round(2))

# Separate Defenders and Midfielders
defenders = population[population["Pos"] == "DF"].copy()
midfielders = population[population["Pos"] == "MF"].copy()
print(f"\nEligible defenders: {len(defenders)}")
print(f"Eligible midfielders: {len(midfielders)}")

# Check sufficient players are available for sampling
if len(defenders) < SAMPLE_SIZE_PER_GROUP:
    raise ValueError("Not enough defenders for the requested sample size.")

if len(midfielders) < SAMPLE_SIZE_PER_GROUP:
    raise ValueError("Not enough midfielders for the requested sample size.")

# A random sample of 150 eligible defenders and 150 eligible midfielders is selected
def_sample_df = defenders.sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)
mid_sample_df = midfielders.sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)


# Combine the two samples
sample = pd.concat([def_sample_df, mid_sample_df], ignore_index=True)

print(f"\nTotal sample size: " f"{len(sample)}")
print("\nSample size by position:")
print(sample["Pos"].value_counts())

# Numerical variables used in the statistical analysis
def_sample = (def_sample_df["Mn/MP"])
mid_sample = (mid_sample_df["Mn/MP"])