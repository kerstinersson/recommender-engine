import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
HANDLE DATA
'''

# read csv file
indata = pd.read_csv('../../Dataset/Systembolaget/sort_sb1.csv')

# filter out wines only
data = indata.copy().loc[indata['Varugrupp'].isin(['Vitt vin', 'Rott vin', 'Mousserande vin'])]

# filter out wines that are no longer in selection
data = data.loc[data['Utgatt'].isin([0])]

# only keep interesting columns
cols_to_keep = ['Artikelid', 'Namn', 'Namn2', 'Varugrupp', 'Varnummer','Prisinklmoms', 'Ursprung', 'Ursprunglandnamn', 'Producent', 'Argang', 'Alkoholhalt', 'Typ', 'RavarorBeskrivning']
df = data[cols_to_keep]

# sort out entries that are empty
df = df[pd.notnull(df.Typ)]

df.to_csv('rev_sysb.csv', encoding='utf-8', index=False)