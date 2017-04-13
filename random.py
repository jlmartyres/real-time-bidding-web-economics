import pandas as pd
from pandas import DataFrame
import numpy as np


#Declare Validation Data Set as "df"
df = pd.read_csv("validation.csv")

#Declare Train Data Set as "df2"
df2 = pd.read_csv("train.csv")


#Getting the Mean Price from the train data set to determine the constant bid
#meanprice = df2["bidprice"].mean()
#print (meanprice)

#Add Columns to the Validation Set with Mean Price
#df["const_bid"] = meanprice



#Create a Column named "random_bid" and add random values
df["1-75"] = np.random.choice(range(1, 75) , df.shape[0])
df["1-150"] = np.random.choice(range(1, 150) , df.shape[0])
df["1-225"] = np.random.choice(range(1, 225) , df.shape[0])
df["1-300"] = np.random.choice(range(1, 300) , df.shape[0])
#Write the output to the csv
df.to_csv("validation.csv", index=False)

#Print
print (df.head())
print (df2.head())

df['CTR'] = df.apply(lambda row: (row['click']*row['impressions']))
df['linear_bid'] = df.apply(lambda row: (200)*(row['pCTR']/row['CTR']),axis=1)