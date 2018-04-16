# content based recommender engine for categorical/text data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

# import utitily functions
from utilities import *
from recommender_cat import *

'''
RECOMMENDER ENGINE
'''

class RecommenderEngine():

	mod_file = './data/sim_mat.csv'

	nr_rec = 20

	def __init__(self):

		sim_csv = pd.read_csv(self.mod_file)

		self.sim_mat = sim_csv.as_matrix(columns = None)

		# read csv file
		data = pd.read_csv('./data/rev_sysb.csv')

		# clean string columns
		features_to_clean = ['Ursprung', 'Producent', 'Typ', 'Varugrupp'] # removed Namn2 and RavarorBeskrivning for special cleaning
		df = data.copy()

		# clean data
		for feature in features_to_clean:
			df[feature] = data[feature].apply(clean_data)

		# create bag of words and clean description box
		df['Namn2'] = df['Namn2'].apply(bag_of_words)
		df['Namn'] = df['Namn'].apply(bag_of_words)
		df['RavarorBeskrivning'] = df['RavarorBeskrivning'].apply(clean_descr)

		# stringify data
		df['keywords'] = df.apply(stringify, axis=1)

		count = CountVectorizer(stop_words='english')
		count_matrix = count.fit_transform(df['keywords'])

		# similarity function
		self.sim = cosine_similarity(count_matrix, count_matrix)

		self.df = df.reset_index()
		self.indices = pd.Series(df.index, index=df['Artikelid']).drop_duplicates()

	# takes article number as input and outputs a list of recommended article numbers
	def recommend(self, art_number):
		# get index of wine
		idx = self.indices[art_number]

		# Get the pairwise similarity scores of all wines with that wine
		sim_scores = list(enumerate(self.sim[idx]))

	    # Sort the wines based on the similarity scores
		self.sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

	    # Get the scores of the 5 most similar wines
		sim_scores = self.sim_scores[1:(self.nr_rec+1)]

	    # Get the movie indices
		wine_indices = [i[0] for i in sim_scores]

		i = 0

		artid = pd.read_csv('./data/artnr_artid.csv')

		# for ind in wine_indices:
		# 	artnr = get_nr(artid, self.df['Artikelid'].iloc[ind])
		# 	print("Recommended: " + str(artnr) + " (score:" + str(sim_scores[i]) + ")")
		# 	i += 1

	    # Return the top 5 most similar wines
		return self.df['Artikelid'].iloc[wine_indices]

	def test_rs(self, nr_test):
		# ten wines to test
		#test = [7904, 7424, 7602, 77152, 5352, 2800]

		artid = pd.read_csv('./data/artnr_artid.csv')

		ids = artid['Artikelid']

		# randomly sample 10 wines to test
		ids = ids.sample(n=nr_test)

		#ids = []

		# for item in test:
		# 	ids.append(get_id(artid, item))

		cov = []
		av_sim = 0
		div = 0
		thres_cov = 0

		# run recommender for each wine
		for wine in ids:
			result = self.recommend(wine)
			score = zip(*self.sim_scores[1:(self.nr_rec+1)])
			#div.append(self.diversity(result.tolist()))
			div += self.diversity(result.tolist())
			#print("-----------")
			thres_cov += self.coverage_thres(wine)
			#av_sim.append(np.mean(score[1]))
			av_sim += np.mean(score[1])
			for item in result: 
				artnr = get_nr(artid, item)
				#res.append(artnr)
				if artnr not in cov:
					cov.append(artnr)

		sim_score = zip(*self.sim_scores)
		#plt.plot(sim_score[1])
		#plt.show()

		tot_cov = self.coverage(cov)
		av_sim = av_sim/nr_test
		av_div = div/nr_test
		av_cov = thres_cov/nr_test

		print("Testvalues for " + str(nr_test) + " items")
		print("Average similarity: ")
		print(av_sim)
		print("-----------")
		print("Coverage: ")
		print(tot_cov)
		print("-----------")
		print("Average items above 0.2 similarity: ")
		print(av_cov)
		print("-----------")
		print("Average diversity: ")
		print(av_div)

		return 0

		# # ten wines to test
		# test = [7904, 7424, 7602, 77152, 5352, 2800]

		# artid = pd.read_csv('./data/artnr_artid.csv')
		# ids = []

		# for item in test:
		# 	ids.append(get_id(artid, item))

		# # run recommender for each wine
		# for wine in ids:
		# 	res = []
		# 	cov = []
		# 	div = []
		# 	av_sim = []
		# 	result = self.recommend(wine)
		# 	score = zip(*self.sim_scores[1:6])
		# 	print("-----------")
		# 	div.append(self.diversity(result.tolist()))
		# 	cov.append(self.coverage(wine))
		# 	av_sim.append(np.mean(score[1]))
		# 	for item in result: 
		# 		res.append(get_nr(artid, item))

		# 	print("For article id: " + str(wine))
		# 	print("Recommendations: ")
		# 	print(res)
		# 	print("Average similarity: ")
		# 	print(av_sim)
		# 	print("-----------")
		# 	print("Coverage: ")
		# 	print(cov)
		# 	print("-----------")
		# 	print("Diversity: ")
		# 	print(div)

		# return 0

	# coverage
	def coverage(self, items):
		rec_items = float(len(items))
		tot_items = float(len(self.indices))
		print(tot_items)
		return rec_items/tot_items

	# calculate coverage
	def coverage_thres(self, artid):
		threshold = 0.2

		# count all scores above a certain value
		above_thres = filter(lambda x: x[1] > threshold, self.sim_scores)

		num_ab = len(above_thres)
		num_tot = len(self.sim_scores)

		#print(num_ab)
		#print(num_tot)
		#print(num_ab/num_tot)

		#res = num_ab/num_tot

		#print(above_thres)
		return float(num_ab)/float(num_tot)

	# calculate diversity
	def diversity(self, artids):
		score = 0
		num = 0
		for i in artids:
			for j in artids:
				if i != j:
					score += 1 - self.sim_mat[self.indices[i], self.indices[j]]
					num += 1

		av_score = score/num # calculate average

		return av_score

	def plot_heat(self):
		data = self.sim
		ax = sns.heatmap(data, xticklabels=500, yticklabels=500)
		#ax = sns.heatmap(data)
		plt.show()

'''
MAIN
'''

if __name__ == '__main__':
	test = False
	nr_rec_test = True
	rs = RecommenderEngine()

	if test:
		nr_test = 1000
		rs.test_rs(nr_test)

	elif nr_rec_test:
		nr_test = 1000
		rs.test_rs(nr_test)

	else:
		#rs.plot_heat()
		artid = pd.read_csv('./data/artnr_artid.csv')
		item = get_id(artid, 2800)
		rs.recommend(item)
# 	rec_input = 1125441 # ripasso della valpolicella
# 	rec = recommend(rec_input)
# 	name = get_info(data, rec_input)
# 	print("Getting recommendations for " + name['Namn'])