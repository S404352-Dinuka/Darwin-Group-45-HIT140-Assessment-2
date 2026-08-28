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

#checking with the dataset information
print("\n Dataset information:")
raw.info()

#checking with the duplicate rows
print("\nDuplicate rows:")
print(raw.duplicated().sum())

#select required variables for the analysis
df = raw[["Player", "Pos", "MP", "Min", "SoTA", "Saves", "Save%"]].copy()
print("\nThe Selected Variables are :")

#checking with the selected variables
print(df.head())
print(df.shape)

#checking with the missing values among selected variables
print("\nMissing values in the selected variables:")
print(df.isnull().sum())

#checking with the players who didn't play the game
print("\nPlayers with zero minutes of play:")
print((df["Min"] == 0).sum())

#Remove the rows which has missing values in Save%
df = df.dropna(subset=["Save%"]).copy()

#checking with the missing values after cleaned save%
print("\nMissing values after cleaning the Save%:")
print(df.isnull().sum())

#checking with the dataset count of rows and columns after cleaning save%
print("\nThe dataset size after cleaning:")
print(df.shape)

# Calculate the shots on target faced per 90 minutes by goalkeepers
df["SoTA_per_90"] = df["SoTA"] / df["Min"] * 90

print("\nSoTA per 90 minutes:")
print(df[["Player", "SoTA", "Min", "SoTA_per_90"]].head())

# Find the median for shots on target per 90 minutes
median_sota = df["SoTA_per_90"].median()

print("\nMedian SoTA per 90 minutes:")
print(median_sota)

# Divide the goalkeepers into lower and higher workloads based on median SoTA per 90 
df["Workload"] = "Lower"
df.loc[df["SoTA_per_90"] > median_sota, "Workload"] = "Higher"

#checking with the number of goalkeepers in each workload group
print("\nNumber of goalkeepers in each workload group:")
print(df["Workload"].value_counts())

#Save the csv file of cleaned data in processed folder
df.to_csv("data/processed/goalkeeping_clean.csv", index=False)

print("\nCleaned data saved.")

# checking with the descriptive statistics for each workload group

print("\nDescriptive statistics:")

print(df.groupby("Workload")["Save%"].describe())


# To visually compare higher and lower workload groups we used boxplot

df.boxplot(column="Save%", by="Workload")

plt.title("Save Percentage by Workload Group")
plt.suptitle("")
plt.xlabel("Workload")
plt.ylabel("Save Percentage")

plt.show()

# 95% confidence intervals for mean Save%
higher = df[df["Workload"] == "Higher"]["Save%"]
lower = df[df["Workload"] == "Lower"]["Save%"]

#checking with the mean save% of higher and lower workload groups
print("\nMean Save%:")
print("Higher workload:", higher.mean())
print("Lower workload:", lower.mean())

# Calculating the 95% confidence interval for mean Save%

higher_ci = stats.t.interval(
    0.95,
    len(higher) - 1,
    loc = higher.mean(),
    scale = stats.sem(higher)
)

lower_ci = stats.t.interval(
    0.95,
    len(lower) - 1,
    loc = lower.mean(),
    scale = stats.sem(lower)
)

print("\n95% Confidence Interval:")
print("Higher workload:", higher_ci)
print("Lower workload:", lower_ci)

# Two-sample t-test
t_test = stats.ttest_ind(higher, lower)

print("\nTwo-sample t-test:")
print("t-statistic:", t_test.statistic)
print("p-value:", t_test.pvalue)
