import pandas as pd
from pandas import DataFrame
import numpy as np
import csv
import os.path

#read data file, must have "ACTR" & "PCTR as columns
df = pd.read_csv("validation.csv")

maxCTRBid = None
maxCTRValue = None

csvFileName = "constant.csv"

if not os.path.isfile(csvFileName):
    thisfile = open(csvFileName, 'w')
    thisfile.write('bid, ctr, clicks, spend, bidtotal, cpm, cpc\n')
    thisfile.close()

for x in range(1, 301):
    df['bid'] = x

    df2 = df[(df.bid > df.payprice)]

    base_bid = x
    ctr = df2['click'].sum().astype('float64') / df2['logtype'].sum().astype('float64')
    clicks = df2['click'].sum()
    spend = df2['bid'].sum()
    bidTotal = df['bid'].sum()
    cpm = spend.astype('float64') / df2['logtype'].sum().astype('float64')
    cpc = None
    if clicks == 0:
        cpc = spend
    else:
        cpc = spend.astype('float64') / clicks

    file = open(csvFileName, 'a')
    file.write('{}, {}, {}, {}, {}, {}, {}\n'.format(base_bid, ctr, clicks, spend, bidTotal, cpm, cpc))
    file.close()

    if ctr > maxCTRValue:
        maxCTRBid = base_bid
        maxCTRValue = ctr

    print "\nBid", x
    print "Click-Through Rate:", ctr
    print "Clicks", clicks
    print "Spend", spend
    print "Total Bidprice:", bidTotal
    print "Average CPM:", cpm
    print "Average CPC:", cpc

    # if bidTotal > 25000000:
    #     print "\nBid Total Exceeded, Terminating"
    #     break

print "\nFinished"
print "\nBase Bid", maxCTRBid, "has largest CTR", maxCTRValue