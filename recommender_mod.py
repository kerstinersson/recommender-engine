# content based recommender engine for categorical/text data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# import utitily functions
from utilities import *
from recommender_mod import *
from similarity_grapes import *

'''
HANDLE DATA
'''

class RecommenderEngine():

	def __init__(self):
		# read csv file
		data = pd.read_csv('./data/rev_sysb.csv')

		self.data = data

		# # clean string columns
		# features_to_clean = ['Ursprung', 'Producent', 'Typ', 'Varugrupp'] # removed Namn2 and RavarorBeskrivning for special cleaning
		# df = data.copy()

		# # clean data
		# for feature in features_to_clean:
		# 	df[feature] = data[feature].apply(clean_data)

		# # create bag of words and clean description box
		# df['Namn2'] = df['Namn2'].apply(bag_of_words)
		# df['Namn'] = df['Namn'].apply(bag_of_words)
		# df['RavarorBeskrivning'] = df['RavarorBeskrivning'].apply(clean_descr)

		# # stringify data
		# df['keywords'] = df.apply(stringify, axis=1)

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
		 	df[feature] = data[feature].apply(clean_data)

		# lower case
		df['Namn2'] = df['Namn2'].apply(bag_of_words)
		df['Namn'] = df['Namn'].apply(bag_of_words)

		# merge name and name2 into one feature
		df['name'] = df.apply(merge_name, axis=1)

		# clean description
		df['RavarorBeskrivning'] = df['RavarorBeskrivning'].apply(clean_descr)

		return df

	def calc_sim(self, wine1, wine2):
		# compare six features: name, group, type, grapes, producer, origin
		total_score = 0
		max_score = 6

		# clean a copy of the dataset
		data = prep_data(self.data.copy()) 

		# calc similarities for all features
		total_score = 

		return 0

	# takes article number as input and outputs a list of recommended article numbers
	def recommend(self, art_number):

		# get variables
		indices = self.indices
		sim = self.sim
		data = self.data

		# get index of wine
		idx = indices[art_number]

		# Get the pairwise similarity scores of all wines with that wine
		sim_scores = list(enumerate(sim[idx]))

	    # Sort the wines based on the similarity scores
		sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

	    # Get the scores of the 5 most similar wines
		sim_scores = sim_scores[1:6]

	    # Get the movie indices
		wine_indices = [i[0] for i in sim_scores]

	    # Return the top 5 most similar wines
		return data['Artikelid'].iloc[wine_indices]

if __name__ == '__main__':
	rs = RecommenderEngine()
	# data = rs.data.copy()
	# new_data = rs.prep_data(data)
	# print(new_data.head(3))
	#recommend(1009797, rs)

