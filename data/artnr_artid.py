import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
HANDLE DATA
'''

# read csv file
indata = pd.read_csv('./rev_sysb.csv')

# only keep interesting columns
cols_to_keep = ['Artikelid', 'Varnummer']
df = indata[cols_to_keep]

df.to_csv('artnr_artid.csv', encoding='utf-8', index=False)