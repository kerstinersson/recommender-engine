import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

'''
UTILITIES
'''

# nothing here yet...

'''
HANDLE DATA
'''

# read csv file
indata = file.read('../Dataset/cellartracker/cellartracker.txt')
for line in indata:
	# read line and export to csv file


# transform data

'''
RECOMMENDER ENGINE
'''

# takes article number as input and outputs a list of recommended article numbers
def recommend(art_number, sim = sim):

	# get index of wine
	idx = indices[art_number]

	# Get the pairwise similarity scores of all movies with that movie
	sim_scores = list(enumerate(sim[idx]))

    # Sort the movies based on the similarity scores
	sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

    # Get the scores of the 10 most similar wines
	sim_scores = sim_scores[1:11]

    # Get the movie indices
	wine_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
	return data['Artikelid'].iloc[wine_indices]

'''
MAIN
'''

if __name__ == '__main__':
	print("running....")