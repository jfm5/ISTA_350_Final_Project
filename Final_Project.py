'''
Author:Jon Moreland
Date: 04/22/2024
Class: ISTA 350
Section Leader: Sara Cielaszyk
Final Project

Description: This code scrapes and visuallizes data in order to showcase how the surface pressure/atmosphere of a planet
plays a much larger role in the mean surface temperature of that planet, than the distance that planet is away from the sun.
'''

import requests 
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from bs4 import BeautifulSoup
from scipy import stats
import matplotlib.pyplot as plt


'''
This function is similar to the one from the homework, it takes in a url that is associated with the desired data table
that I want to be cleaned and set up for me to use. This is the initial function that scrapes the data before I parse
it further.
'''

def get_and_clean_table(url):
    table = pd.read_html(url, index_col = 0, header = 0)[0]
    table = table.transpose()
    table = table.apply(pd.to_numeric, errors = 'coerce')
    table = table.dropna(axis = 1, how = 'all')
    table.columns = table.columns.str.title()
    table_one = table.iloc[:,[0, 1, 3]]
    table_one = table_one.drop(table_one.index[3])
    table_two = table.iloc[:,[7, 15]]
    table_two = table_two.drop(table_two.index[3])
    table_three = table.iloc[[0,1,2,3,4,9],[16, 15]]
    table_three = table_three.drop(table_three.index[3])
    return table_one, table_two, table_three

'''
This function is specifically set up to take the desired values from the input series parameter, and graph
it. It has a parameter for the title of the graph, as well as the y label. The x values specifically need 
a little bit of extra math done in order to properly linearize the data to show the relationship fully, 
which can be seen by the additional math in the x vals. Furthermore this function does a linear regression
to get a line of best fit along with its associated pearson's r and r squared values to further support
the relationship between the data. The pearson's r values are returned so that they can be output with
the data.
'''
def make_grav_plot(series, title, y_label):
    x_vals = (series.iloc[:, 0] * (6.6743*(10**-11))) / ((series.iloc[:,1]/2)**2)
    y_vals = series.iloc[:, 2]
    
    corr, _ = pearsonr(x_vals, y_vals)
    
    m, b = np.polyfit(x_vals, y_vals, 1)
    r_sq = np.corrcoef(x_vals, y_vals)[0, 1]**2  
    

    plt.scatter(x_vals, y_vals, marker='o')
    plt.plot(x_vals, m*x_vals + b, color='red', label=f'Fit: y = {m:.2f}x + {b:.2f}') 
    
    plt.xlabel('Mass (10^24 Kg) / Radius^2 (Km)', size=16)
    plt.ylabel(y_label, size=16)
    plt.title(title, size=20)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return corr

'''
Similarly to the previous function, this one takes in another series, title, and y label.
In this case I already know that the data is not going to be linear so I'm not doing a line 
of best fit, but instead labeling each data point so that it's easier to identify which one
corresponds to which planet. The function calculates and returns a pearsons r to show that 
the data doesn't correlate that well just yet.
'''
def make_distance_plot(series, title, y_label):
    x_vals = series.iloc[:, 0] 
    y_vals = series.iloc[:, 1]
    fig, ax = plt.subplots()
    labels = ['Mercury','Venus','Earth','Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune','Pluto']
    ax.scatter(x_vals,y_vals)
    for i, label in enumerate(labels):
        ax.annotate(label, (x_vals[i],y_vals[i]))
    corr, _ = pearsonr(x_vals, y_vals)
    plt.xlabel('Distance (10^6 Km)', size=16)
    plt.ylabel(y_label, size=16)
    plt.title(title, size=20)
    plt.grid(True)
    plt.show()
    return corr

'''
This function is set up to do a little bit more math with the independent values from the previous
function to show that there is a relation. It does a linear regression based on the series, labels
each data point with it's respective planet, returns a better pearsons r value than the previous
function, and does a linear regression with a line of best fit to further show the outlier in the data.
'''
def make_inv_distance_plot(series, title, y_label):
    x_vals = np.log(series.iloc[:, 0])  # Assuming the first column is Disatance
    y_vals = series.iloc[:, 1]  # Assuming the second column is Mean Temperature
    fig, ax = plt.subplots()
    m, b = np.polyfit(x_vals, y_vals, 1)
    plt.plot(x_vals, m*x_vals + b, color='red', label=f'Fit: y = {m:.2f}x + {b:.2f}')
    labels = ['Mercury','Venus','Earth','Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune','Pluto']
    ax.scatter(x_vals,y_vals)
    for i, label in enumerate(labels):
        ax.annotate(label, (x_vals[i],y_vals[i]))
    corr, _ = pearsonr(x_vals, y_vals)
    plt.xlabel('Distance (10^6 Km)', size=16)
    plt.ylabel(y_label, size=16)
    plt.title(title, size=20)
    plt.legend()
    plt.grid(True)
    plt.show()
    return corr

'''
This function takes the temperature data from the scraped table and makes a bar graph with it.
Since we are going to relate temperature to pressure, and there isn't a lot of pressure data
for the planets in the solar system, we are using a bar chart for singular points of data
that are associated with their planets and labeled as such.
'''
def make_temp_chart(series, title, y_label):
    x_vals = series.iloc[:, 0] 
    y_vals = series.iloc[:, 1] 
    fig, ax = plt.subplots()
    labels = ['Mercury','Venus','Earth','Mars', 'Pluto']
    ax.bar(labels,y_vals, color = 'red')
    plt.ylabel(y_label, size=16)
    plt.title(title, size=20)
    plt.grid(True)
    plt.show()

'''
This function is similar to the previous one, except it is graphing the pressure value from the
table for each planet as opposed to the temperature. 
'''
def make_pressure_chart(series, title, y_label):
    x_vals = series.iloc[:, 0]
    y_vals = series.iloc[:, 1] 
    fig, ax = plt.subplots()
    labels = ['Mercury','Venus','Earth','Mars', 'Pluto']
    ax.bar(labels,x_vals, color = 'blue')
    plt.ylabel(y_label, size=16)
    plt.title(title, size=20)
    plt.grid(True)
    plt.show()

'''
In the main we use all of the previously coded functions to fully display the desired product. We are
using three different tables when making all of the graphs so we first establish each of them and 
their data.

Next we make a scatter plot of the gravity vs mass/r^2 data along with its pearson's r and rsquare value.

Next we make the two graphs that are using the same data for temperature and distance from the Sun, 
but the independent values are calculated a little bit differently. Both return a pearson's r and 
rsquare value to show that the first set isn't linear, but the second set is closer to being linear.

Finally we have the two bar graphs showing the temperature and pressure data for the associated planets
'''
def main():
    table_one, table_two, table_three = get_and_clean_table('https://nssdc.gsfc.nasa.gov/planetary/factsheet/')
    table_one, table_two, table_three
    print(table_one)

    pearson_corr = make_grav_plot(table_one, 'Planetary Gravity vs Mass / Radius^2','Gravity (M/S^2)')

    # Print Pearson's correlation coefficient and r squared
    print("Pearson's r:", pearson_corr)
    print("Pearson's r squared:", pearson_corr**2)

    print(table_two)

    # Creating a scatter plot and calculating Pearson's correlation
    pearson_corr_two = make_distance_plot(table_two, 'Temperature vs Distance From Sun','Temperature(C)')
    inv_pearson_corr_two = make_inv_distance_plot(table_two, 'Temperature vs Log(Distance)', 'Temperature(C)')
    # Print Pearson's correlation coefficient and r squared
    print("Pearson's r:", pearson_corr_two)
    print("Pearson's r squared:", pearson_corr_two**2)
    print("Pearson's r for log(distance):", inv_pearson_corr_two)
    print("Pearson's r squared for log(distance):", inv_pearson_corr_two**2)


    print(table_three)
    pressure_data = table_three.iloc[:,0]
    mean_temp_data = table_three.iloc[:,1]
    print(pressure_data, mean_temp_data)
    t_chart = make_temp_chart(table_three, 'Temperature Chart','Temperature(C)')
    p_chart = make_pressure_chart(table_three, 'Pressure Chart','Pressure (Bars)')

if __name__ == '__main__':
    main()