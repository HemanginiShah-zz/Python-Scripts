# Data analysis with pandas
## Tutorial by sentdex
#    -> https://pythonprogramming.net/data-analysis-python-pandas-tutorial-introduction/
#
#
# Developed by Nathan Shepherd

import pandas as pd
import datetime
from pandas_datareader import data as web
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
'''
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2015, 1, 1)

df = web.DataReader("XOM", 'google', start, end)
print(df.head())

#show graph of data frame (df)
df.plot()
plt.show()
'''

web_stats = {'Day':[1,2,3,4,5,6],
             'Visitors':[43,34,65,56,29,76],
             'Bounce_Rate':[65,67,78,65,45,52]}

df = pd.DataFrame(web_stats)
'''
#print top five rows
print(df.head())
#set index of dataframe
df = df.set_index('Day')
'''
'''
#how to access specific or several specific columns
print(df.Visitors)
print(df['Visitors'])
print(df[['Visitors', 'Bounce_Rate']])
'''
#save dataframe as newfile in currDir
#df.to_csv('newcsv.csv')
#Read in data from csv

#convert column to array sequence
#print(df.Visitors.tolist())

#convert csv to html (saves in currDir)
#df.to_html('example.html')

#convert csv to excel and export
#df.to_excel('example.xls')

#rename specific column headers {'From_str':'To_str'}
df.rename(columns={'Visitors':'Num_Visited'}, inplace=True)

#get all dataframes from url
#fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')


