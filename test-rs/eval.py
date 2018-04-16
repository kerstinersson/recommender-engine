# evaluates the content based recommender engine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import utitily functions
from utilities import *
from recommender_cat import *
from recommender_mod import * 


def test_rs(rs):
	# ten wines to test
	test = [7904, 7424, 7602, 77152, 5352, 2800]

	artid = pd.read_csv('../data/artnr_artid.csv')
	ids = []

	for item in test:
		ids.append(get_id(artid, item))

	# run recommender for each wine
	for wine in ids:
		res = []
		cov = []
		result = rs.recommend(wine)
		cov.append(coverage(rs,wine))
		for item in result: 
			res.append(get_nr(artid, item))
	
	print("Recommendations: ")
	print(res)
	print("-----------")
	print("Coverage: ")
	print(cov)

	return 0

# calculate coverage
def coverage(rs, artid):
	threshold = 0.1

	# get all similarity scores for a certain wine
	sim_scores = rs.get_scores(artid)

	# count all scores above a certain value
	above_thres = sim_scores[numpy.where(sim_scores > threshold)]

	return len(above_thres)/len(sim_scores)

# calculate diversity

if __name__ == '__main__':
	#rs = RecommenderEngine()
	test_rs(rs)
