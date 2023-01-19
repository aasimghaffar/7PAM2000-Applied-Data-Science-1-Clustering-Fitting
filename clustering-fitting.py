#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Created January 2023
# @author: Aasim Ghaffar
"""

import pandas as pd
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet

def read_file(name):
    """
    # this function get data from xlsx
    # then store data in the variable and return the data.
    """
    data = pd.read_csv(name, skiprows = 3)
    data = data.drop(["Unnamed: 66"], axis = 1)
    return data, data.T

def error_ranges(x, func, param, sigma):
    """
    # Calculates the higher and decrease limits for the function, parameters, and sigmas for a single cost or array x. Functions values are calculated for all combos of +/- sigma and the minimal and most are determined.
    # Can be used for all variety of parameters and sigmas >= 1.
    # This movements can be used in project programs.
    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    # list to hold upper and lower limits for parameters
    uplow = []
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y     = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper  

def logistic(t, n0, g, t0):
    """
    # Calculates the logistic feature with scale component n0 and boom charge g
    """
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f
    
def fitting(data, name):
    """
    # This characteristic will separate facts symptoms and then plot diagram exhibit the top information becoming then it will exhibit self assurance of statistics and error top and decrease restrict
    """
    
    # fit exponential growth
    data_for_fitting = pd.DataFrame()
    year             = np.arange(1963, 2020)
    
    print(year)
    
    in_data                   = data[data["Country Name"] == name]
    in_data_fores_area        = in_data[in_data["Indicator Code"] == "EN.ATM.CO2E.LF.KT"]
    in_data_fores_urban       = in_data[in_data["Indicator Code"] == "SP.URB.TOTL"]
    in_data_fores_arable_land = in_data[in_data["Indicator Code"] == "EN.ATM.CO2E.LF.KT"]
    
    in_data_fores_area        = in_data_fores_area.drop(["Country Name", "Indicator Name", "Country Code", "Indicator Code"], axis = 1).T
    in_data_fores_urban       = in_data_fores_urban.drop(["Country Name", "Indicator Name", "Country Code", "Indicator Code"], axis = 1).T
    in_data_fores_arable_land = in_data_fores_arable_land.drop(["Country Name", "Indicator Name", "Country Code", "Indicator Code"], axis = 1).T
    
    in_data_fores_area        = in_data_fores_area.dropna()
    in_data_fores_urban       = in_data_fores_urban.dropna()
    in_data_fores_arable_land = in_data_fores_arable_land.dropna()
        
    data_for_fitting['co2']    = in_data_fores_area
    data_for_fitting['urban']  = in_data_fores_urban
    data_for_fitting['arable'] = in_data_fores_arable_land
    data_for_fitting['Year']   = pd.to_numeric(year)
    popt, covar                = opt.curve_fit(logistic,data_for_fitting['Year'],data_for_fitting['urban'],p0=(2e9, 0.05, 1990.0))
    data_for_fitting["fit"]    = logistic(data_for_fitting["Year"], *popt)
    sigma                      = np.sqrt(np.diag(covar))
    year                       = np.arange(1963, 2040)
    forecast                   = logistic(year, *popt)
    low, up                    = error_ranges(year, logistic, popt, sigma)
    
    data_for_fitting.plot("Year", ["urban", "fit"])
    plt.title(str(name)+" Urban Population Fitting")
    plt.ylabel('Urban') 
    plt.show()
    
    plt.figure()
    plt.plot(data_for_fitting["Year"], data_for_fitting["urban"], label = "Urban")
    plt.title(str(name)+" Urban Population Fitting")
    plt.plot(year, forecast, label = "forecast")
    plt.fill_between(year, low, up, color = "yellow", alpha = 0.7)
    plt.xlabel("Year")
    plt.ylabel("Urban Population")
    plt.legend()
    plt.show()
    
    popt, covar             = opt.curve_fit(logistic,data_for_fitting['Year'], data_for_fitting['co2'], p0 = (2e9, 0.05, 1990.0))
    data_for_fitting["fit"] = logistic(data_for_fitting["Year"], *popt)
    sigma                   = np.sqrt(np.diag(covar))
    forecast                = logistic(year, *popt)
    low, up                 = error_ranges(year, logistic, popt, sigma)
    data_for_fitting.plot("Year", ["co2", "fit"])
    plt.title(str(name)+" Urban Population Fitting")
    plt.ylabel('co2') 
    plt.show()
    
    plt.figure()
    plt.plot(data_for_fitting["Year"], data_for_fitting["co2"], label = "co2")
    plt.title(str(name)+" Urban Population Fitting")
    plt.plot(year, forecast, label = "Forecast")
    plt.fill_between(year, low, up, color = "yellow", alpha = 0.7)
    plt.xlabel("Year")
    plt.ylabel("Urban Population")
    plt.legend()
    plt.show()
    
    popt, covar             = opt.curve_fit(logistic, data_for_fitting['Year'], data_for_fitting['arable'], p0 = (2e9, 0.05, 1990.0))
    data_for_fitting["fit"] = logistic(data_for_fitting["Year"], *popt)
    sigma                   = np.sqrt(np.diag(covar))
    forecast                = logistic(year, *popt)
    low, up                 = error_ranges(year, logistic, popt, sigma)
    data_for_fitting.plot("Year", ["arable", "fit"])
    plt.title(str(name)+" Urban Population Fitting")
    plt.ylabel('Arable') 
    plt.show()
    
    plt.figure()
    plt.plot(data_for_fitting["Year"], data_for_fitting["arable"], label = "Arable")
    plt.title(str(name)+" Urban Population Fitting")
    plt.plot(year, forecast, label = "Forecast")
    plt.fill_between(year, low, up, color = "yellow", alpha = 0.7)
    plt.xlabel("Year")
    plt.ylabel("Arable Land")
    plt.legend()
    plt.show()
    
    return data_for_fitting
    
def k_means_clustring(data, xlabel, ylabel):
    """
    # This characteristic will exhibit the assessment of specific capacity K-means cluster we we used specific statistical techniques and different equipment
    """
    
    df_ex = data[["co2", "arable"]].copy()

    # min and max operate column by column by default
    max_val = df_ex.max()
    min_val = df_ex.min()
    df_ex   = (df_ex - min_val) / (max_val - min_val)

    # set up the clusterer for number of clusters
    ncluster = 3
    kmeans   = cluster.KMeans(n_clusters = ncluster)
    
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_ex) # fit done on x,y pairs
    labels = kmeans.labels_
    
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    print(cen)
    
    # calculate the silhoutte score
    print(skmet.silhouette_score(df_ex, labels))
    
    # plot using the labels to select colour
    plt.figure(figsize = (10.0, 10.0))
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for l in range(ncluster): # loop over the different labels
        plt.plot(df_ex[labels==l]["co2"], df_ex[labels==l]["arable"], \
                 "o", markersize=3, color=col[l])

    # show cluster centres
    for ic in range(ncluster):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize = 10, label = "Cluster "+str(ic))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title("Indicate Cluster of Data")
    plt.show()
    
    df_ex   = data[["co2", "urban"]].copy()

    # min and max operate column by column by default
    max_val = df_ex.max()
    min_val = df_ex.min()
    df_ex   = (df_ex - min_val) / (max_val - min_val)

    # set up the clusterer for number of clusters
    ncluster = 3
    kmeans   = cluster.KMeans(n_clusters=ncluster)
    
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_ex) # fit done on x,y pairs
    labels = kmeans.labels_

    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    print(cen)

    # calculate the silhoutte score
    print(skmet.silhouette_score(df_ex, labels))

    # plot using the labels to select colour
    plt.figure(figsize = (10.0, 10.0))
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    
    # loop over the different labels
    for l in range(ncluster):
        plt.plot(df_ex[labels==l]["co2"], df_ex[labels==l]["urban"], \
                 "o", markersize = 3, color = col[l])

    # show cluster centres
    for ic in range(ncluster):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize = 10, label = "Cluster "+str(ic))

    plt.xlabel(xlabel)
    plt.ylabel("Urban")
    plt.legend()
    plt.title("Indicate Cluster of Data")
    plt.show()
    
if __name__ == "__main__":  
    data,transposed_data = read_file("clustering-fitting.csv")
    
    filter = fitting(data, "China")
    k_means_clustring(filter, "co2", "Arable")
    
    filter = fitting(data, "India")
    k_means_clustring(filter, "co2", "Forest")