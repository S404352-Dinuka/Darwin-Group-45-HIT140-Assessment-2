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
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

##### DATA LOADING #####

# Reproducibility and statistical settings
RANDOM_SEED = 42
SAMPLE_SIZE_PER_GROUP = 150
ALPHA = 0.05
SECTION_WIDTH = 70

# 95% confidence level when alpha = 0.05
CONFIDENCE_LEVEL = 1 - ALPHA

# Raw dataset path
RAW_PATH = "../data/raw/world_cup_2026_player_involvement_raw.csv"

# Load dataset
raw = pd.read_csv(RAW_PATH)

print("=" * SECTION_WIDTH)
print("RAW DATA LOADED")
print("=" * SECTION_WIDTH)

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

print("\n" + "=" * SECTION_WIDTH)
print("DATA WRANGLING AND CLEANING")
print("=" * SECTION_WIDTH)


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
print("=" * SECTION_WIDTH)
print("POPULATION AND SAMPLING")
print("=" * SECTION_WIDTH)


# Define the eligible population
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

# Compare population and sample means
sampling_check = pd.DataFrame(
    {"Population Mean": [defenders["Mn/MP"].mean(), midfielders["Mn/MP"].mean()],
    "Sample Mean": [
            def_sample.mean(),
            mid_sample.mean()
        ]
    },
    index=[
        "Defenders (DF)",
        "Midfielders (MF)"
    ]
)
print("\nPopulation vs sample means:")
print(sampling_check.round(2))
sample.to_csv("../data/processed/world_cup_2026_playing_time_sample.csv", index=False)
print("\nSample saved as: world_cup_2026_playing_time_sample.csv")

##### DESCRIPTIVE STATISTICS #####

print("\n" + "=" * SECTION_WIDTH)
print("DESCRIPTIVE STATISTICS")
print("=" * SECTION_WIDTH)

def descriptive_statistics(values):
    """
    Calculate descriptive statistics for one numerical sample
    """

    q1 = values.quantile(
        0.25
    )

    q3 = values.quantile(
        0.75
    )

    return {

        "Count":
            len(values),

        "Mean":
            values.mean(),

        "Median":
            values.median(),

        "Standard Deviation":
            values.std(),

        "Minimum":
            values.min(),

        "Q1":
            q1,

        "Q3":
            q3,

        "IQR":
            q3 - q1,

        "Maximum":
            values.max(),

        "Range":
            values.max()
            -
            values.min()
    }


# Calculate descriptive statistics
def_stats = descriptive_statistics(
    def_sample
)
mid_stats = descriptive_statistics(
    mid_sample
)

# Create descriptive statistics table
descriptive_table = pd.DataFrame(
    {
        "Defenders (DF)":
            def_stats,

        "Midfielders (MF)":
            mid_stats
    }
)

print("\nDescriptive statistics:")
print(descriptive_table.round(2))

# Save descriptive statistics
descriptive_table.round(3).to_csv("../outputs/descriptive_statistics.csv")

# using the 1.5 x IQR rule to identify possible outliers
print("\nPossible outliers using the 1.5 x IQR rule:")
for label, values in [("Defenders", def_sample), ("Midfielders", mid_sample)]:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = (q3 - q1)
    lower_bound = (q1 - 1.5 * iqr)
    upper_bound = (q3 + 1.5 * iqr)
    outliers = values[(values < lower_bound) | (values > upper_bound)]
    print(f"\n{label}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of possible outliers: {len(outliers)}")

##### VISUALISATION #####

print("\n" + "=" * SECTION_WIDTH)
print("VISUALISATION")
print("=" * SECTION_WIDTH)


#generated histogram of minutes played per appearance for defenders and midfielders
BIN_WIDTH = 10
max_value = max(def_sample.max(), mid_sample.max())
bins = np.arange(0, max_value + BIN_WIDTH, BIN_WIDTH)

plt.figure(figsize=(8, 5))
plt.hist(def_sample, bins=bins, alpha=0.6, label="Defenders (DF)", edgecolor="black")
plt.hist(mid_sample, bins=bins, alpha=0.6, label="Midfielders (MF)", edgecolor="black")
plt.title("Distribution of Minutes Played per Appearance")
plt.xlabel("Minutes played per appearance (Mn/MP)")
plt.ylabel("Number of players")
plt.legend()
plt.grid(axis="y",alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/playing_time_histogram.png", dpi=300)
plt.close()

print("Histogram saved as: " + "playing_time_histogram.png")

##### 95% CONFIDENCE INTERVALS #####

print("\n" + "=" * SECTION_WIDTH)
print("95% CONFIDENCE INTERVALS")
print("=" * SECTION_WIDTH)

def mean_confidence_interval(values, confidence=0.95):
    """
    Calculate a confidence interval for a population mean.
    """
    n = len(values)
    sample_mean = values.mean()
    sample_std = values.std()
    # Standard error
    standard_error = (sample_std / np.sqrt(n))

    # z critical value
    z_critical = stats.norm.ppf((1 + confidence) / 2)

    # Margin of error
    margin_error = (z_critical * standard_error)

    # Confidence interval
    lower_bound = (sample_mean - margin_error)

    upper_bound = (sample_mean + margin_error)

    return (sample_mean, standard_error, lower_bound, upper_bound)

#Defender confidence interval
(def_mean, def_se, def_ci_lower, def_ci_upper) = mean_confidence_interval(def_sample, CONFIDENCE_LEVEL)

# Midfielder confidence interval
(mid_mean, mid_se, mid_ci_lower, mid_ci_upper) = mean_confidence_interval(mid_sample, CONFIDENCE_LEVEL)

print("\nDefenders:")
print(f"Mean = {def_mean:.2f} minutes")
print(f"Standard Error = {def_se:.2f}")
print(f"95% CI = [{def_ci_lower:.2f}, {def_ci_upper:.2f}]")
print("\nMidfielders:")
print(f"Mean = {mid_mean:.2f} minutes")
print(f"Standard Error = {mid_se:.2f}")
print(f"95% CI = [{mid_ci_lower:.2f}, {mid_ci_upper:.2f}]")

# Create confidence interval table
confidence_table = pd.DataFrame(
    {
        "Group": [
            "Defenders (DF)",
            "Midfielders (MF)"
        ],
        "Sample Size": [
            len(def_sample),
            len(mid_sample)
        ],
        "Mean": [
            def_mean,
            mid_mean
        ],
        "Standard Error": [
            def_se,
            mid_se
        ],
        "95% CI Lower": [
            def_ci_lower,
            mid_ci_lower
        ],
        "95% CI Upper": [
            def_ci_upper,
            mid_ci_upper
        ]
    }
)

# Save confidence interval results
confidence_table.round(3).to_csv("../outputs/confidence_intervals.csv", index=False)

##### INDEPENDENT TWO-SAMPLE t-TEST #####

print("\n" + "=" * SECTION_WIDTH)
print("TWO-SAMPLE t-TEST")
print("=" * SECTION_WIDTH)


#Perform independent two-sample t-test
t_statistic, p_value = stats.ttest_ind( def_sample,mid_sample, equal_var=False)


print("\nTwo-sample t-test results:")
print(f"Defender mean: {def_sample.mean():.2f}")
print(f"Midfielder mean: {mid_sample.mean():.2f}")

mean_difference = def_sample.mean() - mid_sample.mean()

print(f"Difference in sample means (DF - MF): {mean_difference:.2f} minutes")
print(f"t statistic: {t_statistic:.4f}")

# Report very small p-values as p < 0.001 rather than 0.000000
if p_value < 0.001:
    print("p-value: < 0.001")
else:
    print(f"p-value: {p_value:.4f}")

print(f"Significance level: {ALPHA}")


# Statistical decision
if p_value < ALPHA:
    decision = ("Reject H0")
    interpretation = ("There is statistically significant evidence of a difference "
        "in average minutes played per appearance between defenders and midfielders."
    )
else:
    decision = ("Fail to reject H0")
    interpretation = (
        "There is insufficient statistical evidence of a difference "
        "in average minutes played per appearance between defenders "
        "and midfielders."
    )


print(f"\nDecision: {decision}")
print(f"Interpretation: {interpretation}")

# Save t-test results
ttest_results = pd.DataFrame(
    [
        {
            "Test":"Independent two-sample t-test",
            "Mean Defenders": round(def_sample.mean(), 3),
            "Mean Midfielders": round(mid_sample.mean(), 3),
            "Mean Difference DF-MF":round(mean_difference, 3),
            "t Statistic": round( t_statistic, 4),
            "p Value": p_value,
            "Alpha": ALPHA,
            "Decision": decision
        }
    ]
)

ttest_results.to_csv(
    "../outputs/ttest_results.csv",
    index=False
)

##### RESULTS SUMMARY #####

print("\n" + "=" * SECTION_WIDTH)
print("RESULTS SUMMARY")
print("=" * SECTION_WIDTH)

summary_table = pd.DataFrame(
    {
        "Statistic": [
            "Sample size",
            "Mean Mn/MP",
            "Median Mn/MP",
            "Standard deviation",
            "95% CI lower",
            "95% CI upper"
        ],

        "Defenders (DF)": [
            len(def_sample),
            round(def_sample.mean(), 2),
            round(def_sample.median(), 2),
            round(def_sample.std(), 2),
            round(def_ci_lower, 2),
            round(def_ci_upper, 2)
        ],

        "Midfielders (MF)": [
            len(mid_sample),
            round(mid_sample.mean(), 2),
            round(mid_sample.median(), 2),
            round(mid_sample.std(), 2),
            round(mid_ci_lower, 2),
            round(mid_ci_upper, 2)
        ]
    }
)

print("\nSummary table:")
print(summary_table.to_string(index=False))
print("\nStatistical comparison:")
print(f"Difference in sample means (DF - MF): {mean_difference:.2f} minutes")
print(f"t statistic: {t_statistic:.4f}")

if p_value < 0.001:
    print("p-value: < 0.001")
else:
    print(f"p-value: {p_value:.4f}")
print(f"Decision: {decision}")


# Save summary
summary_table.to_csv("../outputs/summary_table.csv", index=False)

print("\n" + "=" * SECTION_WIDTH)
print("ANALYSIS COMPLETE")
print("=" * SECTION_WIDTH)

print("\nFiles generated:")
print("1. world_cup_2026_playing_time_clean.csv")
print("2. world_cup_2026_playing_time_sample.csv")
print("3. descriptive_statistics.csv")
print("4. confidence_intervals.csv")
print("5. ttest_results.csv")
print("6. summary_table.csv")
print("7. playing_time_histogram.png")