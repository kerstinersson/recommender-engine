import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
data = pd.read_csv('../Dataset/wine-reviews/winemag-data-130k-v2.csv')

# remove duplicates, removes about 10K entries
data = data.drop_duplicates('description')

print(data.shape)

