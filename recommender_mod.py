# content based recommender engine for categorical/text data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import sys

# import utitily functions
from utilities import *
from similarity_grapes import *
from similarity_year import *
from similarity_regions import * 

'''
HANDLE DATA
'''

class RecommenderEngine():

	data_file = './data/rev_sysb.csv'
	year_file = './data/year.csv'
	sim_file = './data/sim_mat.csv'

	def __init__(self):
		# read csv file
		data = pd.read_csv(self.data_file)
		sim = pd.read_csv(self.sim_file)

		self.sim_mat = sim.as_matrix(columns = None)

		self.clean_data, self.sim = self.prep_data(data.copy())

		self.indices = self.get_indices()
		self.sim_mat = self.sim_matrix()

		# store similarity matrix
		df = pd.DataFrame(self.sim_mat)
		df.to_csv('./data/sim_mat.csv', encoding='utf-8', index=False)

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

		# assemble similarity matrix
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
		score_mod = 0
		max_score = 1 # 1 features are always present

		indices = self.indices
		data = self.clean_data 

		# data set on vintages
		year_data = pd.read_csv(self.year_file, sep=";")
		year_data['Region'] = year_data['Region'].apply(clean_data)

		# calc similarities for all features
		# text features: typ, producent, namn
		idx1 = indices[wine1]
		idx2 = indices[wine2]
		total_score += self.sim[idx1][idx2]*0.5

		# RavarorBeskrivning
		d1 = data[data['Artikelid'] == wine1]['RavarorBeskrivning'].to_string(index = False)
		d2 = data[data['Artikelid'] == wine2]['RavarorBeskrivning'].to_string(index = False)

		# checks similarity if both wines have descriptions
		if d1 != "" and d2 != "":
			score_mod += sim_wines(d1,d2)
			max_score += 1

		# Ursprung
		reg1 = data[data['Artikelid'] == wine1]['Ursprung'].to_string(index = False)
		reg2 = data[data['Artikelid'] == wine2]['Ursprung'].to_string(index = False)

		score = sim_regions(reg1, reg2)

		# adds to total score if regions are in list
		if score != False:
			score_mod += score
			max_score += 1

		#.astype(float)
		# Ar, only taken into account if both vintages are available
		y1 = str(data[data['Artikelid'] == wine1]['Argang'].iloc[0]) #apply(lambda x: "{:.0f}".format(x)) #to_string(index = False)
		y2 = str(data[data['Artikelid'] == wine2]['Argang'].iloc[0]) #apply(lambda x: "{:.0f}".format(x)) #to_string(index = False)

		#print(y1)
		#print(y2)

		if reg1 != "" and reg2 != "":
			simyear = sim_years(y1, y2, reg1, reg2, year_data)

			if simyear != 0:
				score_mod += simyear
				max_score += 1 # add one more feature to add to max score

		# normalize score
		norm_score = (total_score + score_mod*1.5)/max_score

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
		sim_scores = list(enumerate(sim[idx]))

		# Sort the wines based on the similarity scores
		sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

		# Get the scores of the 5 most similar wines
		sim_scores = sim_scores[1:6]

		# Get the wine indices
		wine_indices = [i[0] for i in sim_scores]

		#i = 0

		# for ind in wine_indices:
		# 	print("Recommended: " + str(data['Artikelid'].iloc[ind]) + " (score:" + str(sim_scores[i]) + ")")
		# 	i += 1

		# Return the top 5 most similar wines
		return data['Artikelid'].iloc[wine_indices]

	def get_scores(self, art_number):
		# get variables
		indices = self.indices
		sim = self.sim_mat
		data = self.clean_data

		# get index of wine
		idx = indices[art_number]

		# Get the pairwise similarity scores of all wines with that wine
		sim_scores = list(enumerate(sim[idx]))

		# Sort the wines based on the similarity scores
		sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

		return sim_scores


	def test_rs(self):
		# ten wines to test
		test = [7904, 7424, 7602, 77152, 5352, 2800]

		artid = pd.read_csv('./data/artnr_artid.csv')
		ids = []

		for item in test:
			ids.append(get_id(artid, item))

		# run recommender for each wine
		for wine in ids:
			res = []
			cov = []
			div = []
			av_sim = []
			result = self.recommend(wine)
			sim_score = self.get_scores(wine)
			score = zip(*sim_score[1:6])
			div.append(self.diversity(result.tolist()))
			print("-----------")
			cov.append(self.coverage(wine))
			av_sim.append(np.mean(score[1]))
			for item in result: 
				res.append(get_nr(artid, item))
		
			print("For article id: " + str(wine))
			print("Recommendations: ")
			print(res)
			print("Average similarity: ")
			print(av_sim)
			print("-----------")
			print("Coverage: ")
			print(cov)
			print("-----------")
			print("Diversity: ")
			print(div)

		return 0

	# calculate coverage
	def coverage(self, artid):
		threshold = 0.29

		# get all similarity scores for a certain wine
		sim_scores = self.get_scores(artid)

		#a = sim_scores[1,:] > threshold

		# count all scores above a certain value
		above_thres = filter(lambda x: x[1] > threshold, sim_scores)

		num_ab = len(above_thres)
		num_tot = len(sim_scores)
		return float(num_ab)/float(num_tot)

	# calculate diversity
	def diversity(self, artids):
		score = 0
		num = 0
		for i in artids:
			for j in artids:
				if i != j:
					score += self.sim_mat[self.indices[i], self.indices[j]]
					num += 1

		av_score = score/num # calculate average

		return av_score


if __name__ == '__main__':
	rs = RecommenderEngine()
	#rs.test_rs()

