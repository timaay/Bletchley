import os

cleanfolder = r"C:\Users\timge\Documents\Learning\Machine Learning\Bletchley Bootcamp\Robeco Case\Data (Robeco)\filings_clean_withscore"
rawfolder = r"C:\Users\timge\Documents\Learning\Machine Learning\Bletchley Bootcamp\Robeco Case\Data (Robeco)\filings_raw_withscore"

Cleanedfilelist = os.listdir(cleanfolder)

Rawfilelist = os.listdir(rawfolder)

for filename in Rawfilelist:
    if filename in Cleanedfilelist:
        print(f"This file matches: {filename}")
        os.remove(rawfolder + "\\" + filename)
