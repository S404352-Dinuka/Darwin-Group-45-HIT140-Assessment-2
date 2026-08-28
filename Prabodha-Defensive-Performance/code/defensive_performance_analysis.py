# Analytic Task: Defensive Performance – Prabodha

# Analytical Question: Do defenders and midfielders have a significant difference in interceptions per 90 minutes during FIFA World Cup 2026?

import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats

# Create Outputs Folder for all the analysis and vizualize data storing.
results_folder = os.path.join(os.path.dirname(__file__), "../outputs")
os.makedirs(results_folder,exist_ok=True)

# Load and Read the FIFA World Cup Defensive Dataset
dataset_path = os.path.join(os.path.dirname(__file__),"../data/raw/world_cup_2026_defensive_actions_raw.csv")
world_cup_defensive_data = pd.read_csv(dataset_path)

# Initial Dataset Exploration
print(world_cup_defensive_data.head())

print("Dataset size:", world_cup_defensive_data.shape)
print("Dataset columns:",world_cup_defensive_data.columns.tolist())

world_cup_defensive_data.info()

# Variables selected for the analysis:
"""Player = player name
Pos = playing position
Squad = national team
90s = playing time in 90-minute units
Int = number of interceptions"""

selected_player_data = world_cup_defensive_data[
    [
        "Player",
        "Pos",
        "Squad",
        "90s",
        "Int"
    ]
].copy()

print( "\nMissing values:")
print(selected_player_data.isnull().sum())

# Data Cleaning
# Change numeric variables to numeric format and invalid data is changed to missing data.
selected_player_data["90s"] = pd.to_numeric(selected_player_data["90s"],errors="coerce")
selected_player_data["Int"] = pd.to_numeric(selected_player_data["Int"],errors="coerce")

# Select Eligible Players
"""
Population--> All FIFA World Cup 2026 defenders and midfielders.
Sample --> Players who: Played as DF or MF, at least played one full match and have valid interception data
"""
eligible_players = selected_player_data[
    (selected_player_data["Pos"].isin(["DF", "MF"])) &
    (selected_player_data["90s"] >= 1.0) &
    (selected_player_data["Int"].notna())
].copy()

print("\nNumber of players by position:")
print(eligible_players["Pos"].value_counts())

# Data Preparation –> calculate interceptions per 90 minutes. This makes the comparison fair since the players play different numbers of minutes.
eligible_players["interceptions_per_90"] = (eligible_players["Int"] /eligible_players["90s"])

print("\nPrepared dataset:")
print(eligible_players.head())

# Save Processed Dataset
processed_data_path = os.path.join(results_folder,"eligible_defensive_players.csv")
eligible_players.to_csv(processed_data_path,index=False)
print("Processed dataset saved:",processed_data_path)

# Descriptive Statistics
print("\nOverall statistics for interceptions per 90 minutes:")

overall_statistics = (eligible_players["interceptions_per_90"].describe())
print(overall_statistics)

position_performance_summary = (
    eligible_players
    .groupby("Pos")["interceptions_per_90"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std"
        ]
    )
)

print("\nBetween defenders and midfielders performance comparison :")
print(position_performance_summary)
# Save descriptive statistics
position_performance_summary.to_csv(os.path.join(results_folder,"defensive_descriptive_statistics.csv"))
overall_statistics.to_csv(os.path.join(results_folder, "overall_descriptive_statistics.csv"))

# Confidence Interval
# Estimate the range where the true population mean of interceptions per 90 minutes is likely to exist.
average_interceptions = (eligible_players["interceptions_per_90"].mean())
interception_standard_deviation = (eligible_players["interceptions_per_90"].std())

number_of_players = len( eligible_players)
mean_standard_error = (interception_standard_deviation /(number_of_players ** 0.5))
confidence_level = 0.95
critical_t_value = stats.t.ppf((1 + confidence_level) / 2, number_of_players - 1)

confidence_margin = (critical_t_value * mean_standard_error)
confidence_lower_bound = (average_interceptions -confidence_margin)
confidence_upper_bound = (average_interceptions +confidence_margin)

print("\n95% Confidence Interval:")
print("Lower bound:",round(confidence_lower_bound, 3))
print("Upper bound:",round(confidence_upper_bound, 3))

confidence_results = pd.DataFrame(
    {
        "Average Interceptions per 90": [
            average_interceptions
        ],
        "Lower Bound": [
            confidence_lower_bound
        ],
        "Upper Bound": [
            confidence_upper_bound
        ], 
        "Confidence Level": ["95%"
        ]
    }
)

confidence_results.to_csv(os.path.join(results_folder,"confidence_interval_results.csv"),index=False)

"""Two-Sample t-Test --> Hypothesis:
H0:
There is no significant difference between defenders and midfielders in interceptions per 90 minutes.
H1:
here is a significant difference between defenders and midfielders in interceptions per 90 minutes. """

defender_interceptions = eligible_players[ eligible_players["Pos"] == "DF"]["interceptions_per_90"]
midfielder_interceptions = eligible_players[eligible_players["Pos"] == "MF"]["interceptions_per_90"]

t_test_statistic, t_test_p_value = stats.ttest_ind(defender_interceptions,midfielder_interceptions,equal_var=False)

print("\nTwo-sample t-test results:")
print("T-statistic:",round(t_test_statistic, 3))
print("P-value:",round(t_test_p_value, 4))

if t_test_p_value < 0.05:
    test_result = ("There is a statistically significant difference between defenders and midfielders.")
else:
    test_result = ("There is no statistically significant difference between defenders and midfielders.")

print(test_result)

t_test_results = pd.DataFrame({"T-statistic": [t_test_statistic],"P-value": [t_test_p_value],"Result": [test_result ]})
t_test_results.to_csv( os.path.join(results_folder,"t_test_results.csv"),index=False)

# Box Plot Visualisation is applied since it compares the distribution of interception performances between two different groups.
plt.figure( figsize=(7, 5))
plt.boxplot([defender_interceptions, midfielder_interceptions],labels=["DF","MF"])

plt.title("Interceptions per 90 Minutes: Defenders vs Midfielders")
plt.xlabel( "Playing Position")
plt.ylabel( "Interceptions per 90 Minutes")

plot_file_path = os.path.join(results_folder,"defensive_performance_boxplot.png")

plt.savefig( plot_file_path, bbox_inches="tight")

print("Successfully plot saved", plot_file_path)

plt.show()
plt.close()