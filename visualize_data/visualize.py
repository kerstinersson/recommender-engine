# visualize data in dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
import networkx as nx

from pandas.tools.plotting import table
from similarity_grapes import * 

# standard figure size
# plt.figure(figsize=(12,9))
# plt.figure(figsize=(10, 7.5))

# functions for data visualization
# visualizes distribution of wine styles in dataset
def style(data):
	df = data['Typ'].value_counts(sort=False)
	print(df)

	plt.figure(figsize=(12,9))

	# plot chart
	df.plot(kind='pie', y = df, autopct='%.2f', startangle=90, labels = None, legend = True) # labels = data['Typ']

	plt.show()

	return 0

# visualizes types of wines in dataset
def types(data):
	df = data['Varugrupp'].value_counts()
	print(df)

	return 0

# plots a matrix showing NaN-values in data
def null_pattern(data):
	msno.matrix(data).plot()
	plt.show()

	return 0

# grape network
G=nx.Graph()


# read csv file
data = pd.read_csv('../rev_sysb.csv')

if __name__ == '__main__':
	#style(data)
	#types(data)
	null_pattern(data)