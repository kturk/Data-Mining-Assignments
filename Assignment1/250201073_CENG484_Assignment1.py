import datas as datas
import inline as inline
import klearn as klearn
import matplotlib as matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('AmazonSales.csv', thousands=',')      # To fix 13,000 issue
df['Price'] = pd.to_numeric(df.Price, errors='coerce')  # Casting prices to integer
df['Product'] = df['Product'].str.strip()               # To fix whitespace issue

productGroups = df.groupby(by='Product')            # Grouping by product
print(productGroups.Price.sum())                    # Printing the sum for each product

print(df[df.Payment_Type=='Mastercard'].Price.sum())

img1 = plt.scatter(df['Product'], df['Price'])
plt.show(img1)

countryGroups = df.groupby(by='Country')
print(countryGroups['Product'].count())
print('Total number of country is:', countryGroups['Product'].nunique().count())


#------------------------------------------------------------------------------------------#
                                     #Breast Cancer Part#

df2 = pd.read_csv('wdbc.data')
outliers = []

def detect_outlier(data_1):
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    for y in data_1:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outlier_datapoints = detect_outlier(df2['radius_mean'])
print(outlier_datapoints)
outlier_datapoints = detect_outlier(df2['texture_mean'])
print(outlier_datapoints)

img2 = plt.boxplot(df2['radius_mean'])
plt.show(img2)

img3 = plt.boxplot(df2['texture_mean'])
plt.show(img3)