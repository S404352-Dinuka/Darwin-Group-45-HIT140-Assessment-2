import pandas as pd

#Read the data in the csv file
data = pd.read_csv("../data/raw/misc_stats_raw.csv")

#Print the data to see if its correctly displayed
print(data.info())
print(data.head())
print("Rows and Columns:", data.shape)

#Removed repeated Headers
data = data[data["Rk"] != "Rk"]

#created a temp and re read it again without the repeated headers
data.to_csv("../data/processed/temp.csv", index=False)
data = pd.read_csv("../data/processed/temp.csv")

# print("Dataset after removing headers")
# print(data.info())
print("Rows and Columns after removing duplicate headers", data.shape)

#Get the required variables from the dataset
player_data = data[["Player", "Pos", "Squad", "90s", "Fls"]]

print("Check Player Data")
print(player_data.head())

#Remove any rows that are empty
player_data = player_data.dropna()

print("Rows and Columns after removing any blank fields", player_data.shape)

#Store the player data of midfielders and forwards seperately
midf = player_data[player_data["Pos"] == "MF"]
fwd = player_data[player_data["Pos"] == "FW"]

#Remove players who didn't play in the matches
midf = midf[midf["90s"] != 0]
fwd = fwd[fwd["90s"] != 0]

print("Number of MidFielders: ", midf.shape) 
print("Number of Forwards: ", fwd.shape) 

#Get only fouls that are commited per 90 mins
midf["Fouls_90"] = midf["Fls"] / midf["90s"]
fwd["Fouls_90"] = fwd["Fls"] / fwd["90s"]

#Display the amount of midfielders and forwards present after filtering.
print("Number of MidFielders: ", midf.shape) 
print(midf.head())

print("Number of Forwards: ", fwd.shape) 
print(fwd.head())

#Take a randomly selected sample of midfielders and forwards, then assign a random_state to get the same sample pool again.
midf_sample = midf.sample(n=100, random_state=32)
fwd_sample = fwd.sample(n=100, random_state=32)

print("Number of Midfielders chosen for the sample: ", midf_sample.shape)
print("Number of Forwards chosen for the sample: ", fwd_sample.shape)

#Show descriptive stats
print("MidFeilder Fouls per 90 Mins from sample:")
print(midf_sample["Fouls_90"].describe())

print("Forwards Fouls per 90 Mins from sample:")
print(fwd_sample["Fouls_90"].describe())

#Combine both groups into one dataset
final_data = pd.concat([midf, fwd])
final_sample_data = pd.concat([midf_sample, fwd_sample])

#Save the final data as csv
final_data.to_csv("../data/processed/misc_stats_cleaned.csv", index=False)
final_sample_data.to_csv("../data/processed/misc_stats_cleaned_sample.csv", index=False)

print("Cleaned Datasets Saved!")