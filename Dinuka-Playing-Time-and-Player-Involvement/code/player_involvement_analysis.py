import pandas as pd

file_path = "../data/raw/world_cup_2026_player_involvement_raw.csv"

# Load the raw dataset
df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())
