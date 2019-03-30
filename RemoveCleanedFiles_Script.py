import os

#set this path to your folder containing cleaned files
cleanfolder = r"C:\Users\timge\Documents\Learning\Machine Learning\Bletchley Bootcamp\Robeco Case\Data (Robeco)\filings_clean_withscore"

#set this path to your folder containing raw / uncleaned files
rawfolder = r"C:\Users\timge\Documents\Learning\Machine Learning\Bletchley Bootcamp\Robeco Case\Data (Robeco)\filings_raw_withscore"

#Create lists of filenames in both directories
Cleanedfilelist = os.listdir(cleanfolder)

Rawfilelist = os.listdir(rawfolder)

#For each filename in raw, check if it exists in the cleaned folder, if true, delete filenname from raw
for filename in Rawfilelist:
    if filename in Cleanedfilelist:
        print(f"This file matches: {filename}")
        os.remove(rawfolder + "\\" + filename)
