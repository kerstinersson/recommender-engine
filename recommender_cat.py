# content based recommender engine for categorical/text data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# import utitily functions
from utilities import *

'''
HANDLE DATA
'''

# read csv file
data = pd.read_csv('./rev_sysb.csv')

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
sim = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['Artikelid']).drop_duplicates()

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

    # Get the scores of the 5 most similar wines
	sim_scores = sim_scores[1:6]

    # Get the movie indices
	wine_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
	return df['Artikelid'].iloc[wine_indices]

'''
MAIN
'''

if __name__ == '__main__':
	rec_input = 1125441 # ripasso della valpolicella
	rec = recommend(rec_input)
	name = get_info(data, rec_input)
	print("Getting recommendations for " + name['Namn'])

	for item in rec:
		info = get_info(data, item)
		to_print = "Namn: " + info['Namn'] + " " + info['Namn2'] + ", " + "Ursprung: " + info['Ursprung']
		if not info['RavarorBeskrivning'].isnull:
			to_print += ", Ravaror: " + info['RavarorBeskrivning']
		print(to_print)