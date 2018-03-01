import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from similarity_grapes import sim_grapes

from utilities import *

'''
HANDLE DATA
'''

# read csv file
data = pd.read_csv('./rev_sysb.csv')

# clean string columns
features_to_clean = ['Ursprung', 'Producent', 'Typ', 'Varugrupp', 'Namn'] # removed Namn2 and RavarorBeskrivning for special cleaning

df = data.copy()

# clean data
for feature in features_to_clean:
	df[feature] = data[feature].apply(clean_data)

# special cleaning
df['Namn2'] = df['Namn2'].apply(clean2)
df['RavarorBeskrivning'] = df['RavarorBeskrivning'].apply(clean_descr)

# stringify data
df['keywords'] = df.apply(stringify, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['keywords'])

# similarity function
sim = cosine_similarity(count_matrix, count_matrix)

#print(sim)

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

    # Get the scores of the 10 most similar wines
	sim_scores = sim_scores[1:11]

    # Get the movie indices
	wine_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
	return df['Artikelid'].iloc[wine_indices]

'''
MAIN
'''

if __name__ == '__main__':
	rec_input = 1125441 # ripasso della valpolicella
	rec = recommend(rec_input)
	name = get_info(data, rec_input)
	print("Getting recommendations for " + name)

	for item in rec:
		#print(item)
		info = get_info(data, item)
		print(info)