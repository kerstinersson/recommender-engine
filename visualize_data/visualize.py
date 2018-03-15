# visualize data in dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
import networkx as nx
import sys
#import plotly as py
#import plotly.graph_objs as go

sys.path.append('../../recommender-engine/')

from pandas.tools.plotting import table
from pandas.plotting import parallel_coordinates
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

# build network of nodes
def build_network(matrix):

	grapes = get_grapes()

	# grape network
	G=nx.Graph()

	# create edges
	for i in range(len(matrix)):
		grape = grapes[i]
		for j in range(len(matrix)):
			if i != j and matrix[i][j] != 0:
				G.add_edge(grape, grapes[j], weight = matrix[i][j])


	e25 = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] == 0.25]
	e50 = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] == 0.50]
	e75 = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] == 0.75]
	e100 = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] == 1]

	# position for nodes
	pos = nx.spring_layout(G)

	# nodes
	nx.draw_networkx_nodes(G, pos, node_size = 700)

	# edges
	nx.draw_networkx_edges(G, pos, edgelist = e25, width = 2, edge_color = 'y')
	nx.draw_networkx_edges(G, pos, edgelist = e50, width = 2, edge_color = 'b')
	nx.draw_networkx_edges(G, pos, edgelist = e75, width = 2, edge_color = 'r')
	nx.draw_networkx_edges(G, pos, edgelist = e100, width = 2, edge_color = 'g')

	# labels
	nx.draw_networkx_labels(G, pos)

	return G

def parallel_coords(data):
	plt.figure()
	parallel_coordinates(data, 'Namn')

	return 0

def origin(data):
	# data to visualize origin of wines in data set
	df = data['Ursprunglandnamn'].value_counts(sort=True)
	print(df)


# read csv file
data = pd.read_csv('../data/rev_sysb.csv')

if __name__ == '__main__':



	#origin(data)
	#style(data)
	#types(data)
	#null_pattern(data)
	#matrix = mat_grapes()
	#net = build_network(matrix)

	#print(sorted(d for n, d in net.degree()))
	#nx.draw(net, with_labels=False, font_weight='bold')

	#parallel_coords(data)

	#plt.show()

