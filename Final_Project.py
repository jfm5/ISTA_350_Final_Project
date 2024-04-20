'''
Author:Jon Moreland
Date: 04/22/2024
Class: ISTA 350
Section Leader: Sara Cielaszyk
Final Project

EDIT: Description: This code establishes a person class with a bunch of potential information defining a person. 
There is a class method in the code that can parse a beautiful soup object and create a person class
based off the parsed information, and is the main function of the code. The other functions can produce
the information stored in the class and also as well as compare two classes against each other in order
to try and determine if they are the same or in what ways they are different from each other.
'''

import requests 
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from bs4 import BeautifulSoup
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

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

def make_grav_plot(series, title, y_label):
    x_vals = (series.iloc[:, 0]) / ((series.iloc[:,1]/2)**2)  # Assuming the first column is Mass
    y_vals = series.iloc[:, 2]  # Assuming the second column is Gravity
    plt.scatter(x_vals, y_vals, marker='o')
    plt.xlabel('Mass (10^24 Kg) / (Diameter / 2)^2', size=16)
    plt.ylabel(y_label, size=16)
    plt.title(title, size=20)
    plt.grid(True)
    plt.show()

def make_new_grav_plot(series, title, y_label):
    # Assuming the first column is Mass, the second is Diameter, and the third is Gravity
    x_vals = (series.iloc[:, 0]) / ((series.iloc[:,1]/2)**2)
    y_vals = series.iloc[:, 2]
    
    # Calculate Pearson's correlation coefficient
    corr, _ = pearsonr(x_vals, y_vals)
    
    plt.scatter(x_vals, y_vals, marker='o')
    plt.xlabel('Mass (10^24 Kg) / (Diameter / 2)^2', size=16)
    plt.ylabel(y_label, size=16)
    plt.title(title, size=20)
    plt.grid(True)
    plt.show()
    
    return corr 

def make_distance_plot(series, title, y_label):
    x_vals = series.iloc[:, 0]  # Assuming the first column is Disatance
    y_vals = series.iloc[:, 1]  # Assuming the second column is Mean Temperature
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

def make_temp_chart(series, title, y_label):
    x_vals = series.iloc[:, 0]  # Assuming the first column is Disatance
    y_vals = series.iloc[:, 1]  # Assuming the second column is Mean Temperature
    fig, ax = plt.subplots()
    labels = ['Mercury','Venus','Earth','Mars', 'Pluto']
    ax.bar(labels,y_vals, color = 'red')
    plt.ylabel(y_label, size=16)
    plt.title(title, size=20)
    plt.grid(True)
    plt.show()

def make_pressure_chart(series, title, y_label):
    x_vals = series.iloc[:, 0]  # Assuming the first column is Disatance
    y_vals = series.iloc[:, 1]  # Assuming the second column is Mean Temperature
    fig, ax = plt.subplots()
    labels = ['Mercury','Venus','Earth','Mars', 'Pluto']
    ax.bar(labels,x_vals, color = 'blue')
    plt.ylabel(y_label, size=16)
    plt.title(title, size=20)
    plt.grid(True)
    plt.show()

def main():
    table_one, table_two, table_three = get_and_clean_table('https://nssdc.gsfc.nasa.gov/planetary/factsheet/')
    table_one, table_two, table_three
    print(table_one)

    # Creating a scatter plot and calculating Pearson's correlation
    pearson_corr = make_new_grav_plot(table_one, 'Gravity vs Mass / (diameter/2)^2','Gravity')

    # Print Pearson's correlation coefficient and r squared
    print("Pearson's r:", pearson_corr)
    print("Pearson's r squared:", pearson_corr**2)

    print(table_two)

    # Creating a scatter plot and calculating Pearson's correlation
    pearson_corr_two = make_distance_plot(table_two, 'Temperature vs Distance From Sun','Temperature(C)')

    # Print Pearson's correlation coefficient and r squared
    print("Pearson's r:", pearson_corr_two)
    print("Pearson's r squared:", pearson_corr_two**2)


    print(table_three)
    pressure_data = table_three.iloc[:,0]
    mean_temp_data = table_three.iloc[:,1]
    print(pressure_data, mean_temp_data)
    t_chart = make_temp_chart(table_three, 'Temperature Chart','Temperature(C)')
    p_chart = make_pressure_chart(table_three, 'Pressure Chart','Pressure (Bars)')

if __name__ == '__main__':
    main()