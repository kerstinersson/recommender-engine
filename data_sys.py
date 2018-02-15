import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read csv file
data = pd.read_csv('../Dataset/sort_sb1.csv')

# filter out wines only
data = data.loc[data['Varugrupp'].isin(['Vitt vin','Rott vin', 'Mousserande vin'])]

# filter out wines that are no longer in selection
data = data.loc[data['Utgatt'].isin([0])]

# only keep interesting columns
cols_to_keep = ['Artikelid', 'Prisinklmoms', 'Ursprung', 'Producent', 'Argang', 'Alkoholhalt', 'Typ', 'RavarorBeskrivning']
data = data[cols_to_keep]

print(data.shape)

# check for NaN-values in the data structure
print(data.isnull().sum())