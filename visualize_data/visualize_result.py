import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

# file paths
rm_file = '../../Resultat/performance/recs_mod.csv'
rc_file = '../../Resultat/performance/recs_cat.csv'
wm_file = '../../Resultat/performance/weights_mod.csv'

# create dataframes
rm = pd.read_csv(rm_file, sep=";")
rc = pd.read_csv(rc_file, sep=";")
wm = pd.read_csv(wm_file, sep=";")

# edit data
cols_to_drop = [0, 20, 21]
rows_to_drop = range(3,10)
wm = wm.drop(wm.columns[cols_to_drop], axis=1)
wm = wm[:3]

cols_to_drop = [0]
rm = rm.drop(rm.columns[cols_to_drop], axis=1)

cols_to_drop = [0, 5, 6]
rc = rc.drop(rc.columns[cols_to_drop], axis=1)

def plot_weights():
	ax = plt.subplot(111)
	x = np.arange(0.05, 1.0, 0.05) # weights
	legends = ["Average similarity", "Average coverage", "Average diversity"]
	#sns.color_palette("Reds")

	# style plot
	sns.set_style("ticks")
	sns.despine()
	colors = ["#5f0f40", "#ad0a0a", "#cb793a"]

	#plot
	for rows in wm.index:
		plt.plot(x, wm.loc[rows], color=colors[rows])

	plt.legend(legends)
	plt.show()

def plot_recs(mode):
	# style plot
	sns.set_style("ticks")
	sns.despine()
	colors = ["#5f0f40", "#cb793a"]

	if mode == "comp":
		x = [5, 10, 15, 20] # number of recommendations 

		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)

		# plot similarity
		ax1.scatter(x, rc.loc[0], color=colors[0])
		ax1.scatter(x, rm.loc[0], color=colors[1])
		ax1.set_title("Average similarity")

		ax2.scatter(x, rc.loc[1], color=colors[0])
		ax2.scatter(x, rm.loc[1], color=colors[1])
		ax2.set_title("Average coverage")

		ax3.scatter(x, rc.loc[2], color=colors[0], label="Vector representation")
		ax3.scatter(x, rm.loc[2], color=colors[1], label="Knowledge based")
		ax3.set_title("Average diversity")
		
		legends = ["Vector representation", "Knowledge based"]
		handles, labels = ax3.get_legend_handles_labels()
		#fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05))
		plt.legend( handles=handles,loc="lower left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Legend", fancybox=True)
		fig.tight_layout()
		plt.show()
		#plt.savefig("../data/output.png", bbox_inches="tight")

def plot_recommendations(data1, data2):
	# style plot
	sns.set_style("ticks")
	sns.despine()
	colors = ["#5f0f40", "#cb793a"]

	x = [5, 10, 15, 20] # number of recommendations 

	ax = plt.subplot(111)

	plt.scatter(x, data1, color=colors[0])
	plt.scatter(x, data2, color=colors[1])

	legends = ["Vector representation", "Knowledge based"]
	plt.legend(legends)
	plt.xlabel("Number of recommended items per run")
	plt.show()

if __name__ == '__main__':
	#print("hello")
	#print(wm.head(1))
	#plot_weights()
	plot_recommendations(rc.loc[0], rm.loc[0])
	plot_recommendations(rc.loc[1], rm.loc[1])
	plot_recommendations(rc.loc[2], rm.loc[2])
