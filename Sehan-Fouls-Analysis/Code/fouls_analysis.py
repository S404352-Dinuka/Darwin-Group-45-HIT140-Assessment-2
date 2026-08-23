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

#Get the required variables from the dataset
player_data = data[["Player", "Pos", "Squad", "90s", "Fls"]]

#Remove any rows that are empty
player_data = player_data.dropna()

#Store the player data of midfielders and forwards
midf = player_data[player_data["Pos"] == "MF"]
fwd = player_data[player_data["Pos"] == "FW"]

#Remove players who didn't play in the matches
midf = midf[midf["90s"] != "0"]
fwd = fwd[fwd["90s"] != "0"]

#Get only fouls that are commited per 90 mins
midf["Fouls_90"] = midf["Fls"] / midf["90s"]
fwd["Fouls_90"] = fwd["Fls"] / fwd["90s"]

#Display the amount of midfielders and forwards present after filtering.
print("Number of MidFielders: ", midf.shape[0]) 
print(midf.head())

print("Number of Forwards: ", fwd.shape[0]) 
print(fwd.head())

#Show descriptive stats
print("MidFeilder Fouls per 90 Mins:")
print(midf["Fouls_90"].describe())

print("Forwards Fouls per 90 Mins:")
print(fwd["Fouls_90"].describe())

#Combine both groups into one dataset
final_data = pd.concat([midf, fwd])

#Save the final data as csv
final_data.to_csv("../data/processed/misc_stats_cleaned.csv", index=False)