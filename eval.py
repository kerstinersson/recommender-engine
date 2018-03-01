# evaluates the content based recommender engine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import utitily functions
from utilities import *
from recommender_cat import *
from recommender_cat import * 


def test_rs():
	# ten wines to test
	test = [7904, 7424, 7602, 77152, 5352, 2800]

	artid = pd.read_csv('./artnr_artid.csv')
	ids = []

	for item in test:
		ids.append(get_id(artid, item))

	# run recommender for each wine
	for wine in ids:
		res = []
		result = recommend(wine)
		for item in result: 
			res.append(get_nr(artid, item))
		print(res)


if __name__ == '__main__':
	test_rs()
