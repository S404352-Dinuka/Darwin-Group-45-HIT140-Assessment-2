import pandas as pd
import os

file_path = os.path.join(
    os.path.dirname(__file__),
    "../data/raw/world_cup_2026_defensive_actions_raw.csv"
)

df = pd.read_csv(file_path)

print(df.head())

print("Dataset size:", df.shape)
print(df.columns.tolist())
df.info()

print("Total Rows and columns:", df.shape)

analysis_df = df[["Player", "Pos", "Squad", "90s", "Int"]].copy()

analysis_df.head()

analysis_df.isnull().sum()

### Data Cleaning and Eligibility

#Players who were designated as either a defender (DF) or midfielder (MF) only, as well as players who have played for a minimum of 90 minutes, were chosen.

# Convert required columns to numbers
analysis_df["90s"] = pd.to_numeric(analysis_df["90s"], errors="coerce")
analysis_df["Int"] = pd.to_numeric(analysis_df["Int"], errors="coerce")

# Keep only eligible defenders and midfielders
eligible_df = analysis_df[
    (analysis_df["Pos"].isin(["DF", "MF"])) &
    (analysis_df["90s"] >= 1.0) &
    (analysis_df["Int"].notna())
].copy()

print(eligible_df["Pos"].value_counts())