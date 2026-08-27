import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#To read the CSV file
raw = pd.read_csv("data/raw/FIFA_2026_Goalkeeping.csv", header=1)
#checking whether the data has been loaded
print(raw.head())
print(raw.shape)

#checking with the column names for the given data
print("\n Column names:")
print(raw.columns.tolist())

#checking with the missing values in data
print("\n Missing Values:")
print(raw.isnull().sum())

