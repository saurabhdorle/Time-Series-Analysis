# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:50:30 2018

@author: Saurabh
"""
# Import Liberaries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams 
rcParams['figure.figsize'] = 15, 16
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import dataset
data = pd.read_csv("Train.csv")

# Check Data Types
print(data.dtypes)

# Coverting to Datetime format
data['Datetime'] = pd.to_datetime(data.Datetime, format='%d-%m-%Y %H:%M')

# Start and end date for dataset
data['Datetime'].min()
data['Datetime'].max()

# Creating new features of date
'''
for i in data:
    i['year'] = i.Datetime.dt.year
    i['month']=i.Datetime.dt.month 
    i['day']=i.Datetime.dt.day
    i['Hour']=i.Datetime.dt.hour'''
    
data['year'] = data.Datetime.dt.year
data['month']=data.Datetime.dt.month 
data['day']=data.Datetime.dt.day
data['Hour']=data.Datetime.dt.hour

# weekend variable
data['day of week'] = data['Datetime'].dt.dayofweek
temp = data['Datetime']

def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0

temp2 = data['Datetime'].apply(applyer)
data['weekend']=temp2

data.head()

# indexing the Datetime to get the time period on the x-axis
data.index = data['Datetime']   
# drop ID variable to get only the Datetime on x-axis
data.drop('ID', axis = 1, inplace = True) 

# Time Series analysis for Count attribute
ts = data['Count']

# Plot the graph for Count
plt.figure(figsize = (16, 8))
plt.plot(ts, label = 'Passanger Count')
plt.title('Time Series')
plt.xlabel("Time(Year-Month)")
plt.ylabel("Passanger Count")
plt.legend(loc = 'best')

# Year wise count
data.groupby('year')['Count'].mean().plot.bar()
# Month wise Count
data.groupby('month')['Count'].mean().plot.bar()
# Year and Month wise Count
temp = data.groupby(['year','month'])['Count'].mean()
temp.plot(figsize = (15, 5), title = 'Passenger Count(Monthwise)')
# Daywise Count
data.groupby('day')['Count'].mean().plot.bar()

# Hourly passenger count
data.groupby('Hour')['Count'].mean().plot.bar()

# Count on weekend
data.groupby('weekend')['Count'].mean().plot.bar()

# Count by Day of Week
data.groupby('day of week')['Count'].mean().plot.bar()


######

data.Timestamp = pd.to_datetime(data.Datetime, format = '%d-%m-%Y %H:%M')
data.index = data.Timestamp

# Hourly Time Series
hourly = data.resample('H').mean()

daily = data.resample('D').mean()

weekly = data.resample('W').mean()

monthly = data.resample('M').mean()


fig, axs = plt.subplots(4,1)

hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0])
daily.Count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1])
weekly.Count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2])
monthly.Count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3])
plt.show()





# =============================================================================
# Model Building
# =============================================================================

Train = data.ix['2012-08-25':'2014-06-24']
valid = data.ix['2014-06-25':'2014-09-25']

Train.Count.plot(figsize = (15, 8), title = 'Daily Ridership', fontsize = 14, label = 'Train')
valid.Count.plot(figsize = (15, 8), title = 'Daily Ridership', fontsize = 14, label = 'Valid')
plt.xlabel("Datetime")
plt.ylabel("Passenger count")
plt.legend(loc = 'best')
plt.show()


