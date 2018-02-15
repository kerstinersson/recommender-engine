# calculates the similarity between two types of wines

# build tree

def sim_wines(w1, w2):
	if w1 is None or w2 is None:
		print("No type of wine given...")
		return 0
	else:
		print("calculating similarity")
		return 1

if __name__ == '__main__':
	sim_wines(None, "Wow")