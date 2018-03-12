# content based recommender engine for categorical/text data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# import utitily functions
from utilities import *
#from recommender_mod import *
from similarity_grapes import *

'''
HANDLE DATA
'''

class RecommenderEngine():

	data_file = './data/rev_sysb.csv'

	def __init__(self):
		# read csv file
		data = pd.read_csv(self.data_file)

		self.clean_data, self.sim = self.prep_data(data.copy())

		self.indices = self.get_indices()
		self.sim_mat = self.sim_matrix()

		# store similarity matrix
		df = pd.DataFrame(self.sim_mat)
		df.to_csv('sim_mat.csv', encoding='utf-8', index=False)

		# count = CountVectorizer(stop_words='english')
		# self.count_matrix = count.fit_transform(df['keywords'])

		# # similarity function
		# self.sim = cosine_similarity(self.count_matrix, self.count_matrix)

		# self.data = df.reset_index()
		# self.indices = pd.Series(df.index, index=df['Artikelid']).drop_duplicates()

	def prep_data(self, df):
		# lower case and remove spaces
		features_to_clean = ['Ursprung', 'Producent', 'Typ', 'Varugrupp']
		for feature in features_to_clean:
		 	df[feature] = df[feature].apply(clean_data)

		# lower case
		df['Namn2'] = df['Namn2'].apply(bag_of_words)
		df['Namn'] = df['Namn'].apply(bag_of_words)

		# merge name and name2 into one feature
		df['name'] = df.apply(merge_name, axis=1)

		# stringify namn, producent, typ, varugrupp
		df['text'] = df.apply(stringify_mod, axis=1)

		count = CountVectorizer(stop_words='english')
		count_matrix = count.fit_transform(df['text'])

		# similarity function
		sim = cosine_similarity(count_matrix, count_matrix)

		# clean description
		df['RavarorBeskrivning'] = df['RavarorBeskrivning'].apply(clean_descr)

		return df, sim

	def get_indices(self):
		df = self.clean_data

		# store all article IDs
		artIds = df['Artikelid'].values
		ids = range(artIds.size)

		#self.sim_mat = sim_mat
		indices = dict(zip(artIds,ids))

		return indices

	def sim_matrix(self):
		# get data
		df = self.clean_data

		# store all article IDs
		artIds = df['Artikelid'].values
		ids = range(artIds.size)
		d = dict(zip(ids, artIds))
		
		# matrix of zeros
		sim_mat = np.zeros((artIds.size, artIds.size))

		for i in ids:
			sim_mat[i][i] = 1 # diagonal items, always =1!

			for j in range(i):
				sim_mat[i][j] = self.calc_sim(d[i], d[j])
				sim_mat[j][i] = sim_mat[i][j]
				#print(d[i], d[j])
			print(i)

		return sim_mat

	def calc_sim(self, wine1, wine2):
		# compare six features: name, group, type, grapes, producer, origin
		total_score = 0
		max_score = 5

		indices = self.indices
		data = self.clean_data 

		# calc similarities for all features
		# text features: typ, producent, namn
		idx1 = indices[wine1]
		idx2 = indices[wine2]
		total_score += self.sim[idx1][idx2]

		# RavarorBeskrivning
		d1 = data[data['Artikelid'] == wine1]['RavarorBeskrivning'].to_string(index = False)
		d2 = data[data['Artikelid'] == wine2]['RavarorBeskrivning'].to_string(index = False)

		total_score += sim_wines(d1,d2)

		# Ursprung
		reg1 = data[data['Artikelid'] == wine1]['Ursprung'].to_string(index = False)
		reg2 = data[data['Artikelid'] == wine2]['Ursprung'].to_string(index = False)

		# TODO: add function call to similarity_regions

		# normalize score
		norm_score = total_score/max_score

		return norm_score

	# takes article number as input and outputs a list of recommended article numbers
	def recommend(self, art_number):

		# get variables
		indices = self.indices
		sim = self.sim_mat
		data = self.clean_data

		# get index of wine
		idx = indices[art_number]

		# Get the pairwise similarity scores of all wines with that wine
		sim_scores = list(enumerate(sim_mat[idx]))

		# Sort the wines based on the similarity scores
		sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

		# Get the scores of the 5 most similar wines
		sim_scores = sim_scores[1:6]

		# Get the wine indices
		wine_indices = [i[0] for i in sim_scores]

		# Return the top 5 most similar wines
		return data['Artikelid'].iloc[wine_indices]

if __name__ == '__main__':
	rs = RecommenderEngine()
	#wine1 = 1006372
	#wine2 = 1021015
	# print(rs.clean_data.Ursprung.unique())
	# rs.sim_matrix()
	# rs.calc_sim(wine1, wine2)
	# data = rs.data.copy()
	# new_data = rs.prep_data(data)
	# print(new_data.head(3))
	#print(recommend(1009797))

