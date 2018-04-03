import pandas as pd
import numpy as np

# calculates the similarity between years
# NOTE: requires that the wines are from the same region

def sim_years(y1, y2, region1, region2, x):
	if region1 in x.Region.values and region2 in x.Region.values:
		# region is in data set!

		if y1 in x.columns and y2 in x.columns:
			# years in data set!
			score1 = x[x['Region'] == region1][y1]
			score2 = x[x['Region'] == region2][y2]

			#x.loc[x['Varnummer'] == artNr, 'Artikelid'].iloc[0]

			print(score1)
			print(score2)

			# if np.isnan(score1) or np.isnan(score2):
			# 	return 0
			# else: 
			sim = 1 - abs(score1-score2)/5

			return sim
		else: 
			return 0
	else:
		# region not in list, return a mean
		return 0


# if __name__ == '__main__':
#  	df = pd.read_csv('./data/year.csv', sep=";")
#  	y1 = '2015'
#  	y2 = '2008'
#  	region1 = region2 = 'Champagne'
# 	print(sim_years(y1, y2, region1, region2, df))