# utility functions used in the recommender engines

'''
UTILITIES
'''

# clean data
def clean_data(x):
	if isinstance(x, str):
		return str.lower(x.replace(" ", "")) # lower case only and remove spaces
	else:
		return ''

# bag of words, turn NaN into string
def bag_of_words(x):
	if isinstance(x,str):
		return str.lower(x)
	else:
		return ''

# returns cleaned version of description box
def clean_descr(x):
	if isinstance(x, str):
		# remove percentages
		res = filter(lambda a: a.isalpha() or a == " ", x)

		# write cabernets and pinots as one word to avoid confusion, remove stop words
		repls = ("pinot ", "pinot"), ("cabernet ", "cabernet"), ("och", ""), ("touriga ", "touriga"), ("nero d ", "nerod"), ("chenin ", "chenin"), ("gruner ", "gruner"), ("melon de ", "melonde"), ("petit ", "petit"), ("sauvignon ", "sauvignon"), ("tinta ", "tinta")
		return reduce(lambda a, kv: a.replace(*kv), repls, res)
	else: 
		return ''

# make string with keywords
def stringify(x):
	#return x['Ursprung'] + ' ' + x['Producent'] + ' ' + x['Typ'] + ' ' + x['RavarorBeskrivning'] + ' ' + x['Varugrupp'] + ' ' + x['Namn'] + ' ' + x['Namn2']
	return x['Ursprung'] + ' ' + x['Typ'] + ' ' + x['RavarorBeskrivning'] + ' ' + x['Varugrupp'] + ' ' + x['Namn'] + ' ' + x['Namn2']

# return info on item from article id
def get_info(x, artId):
	return x[x['Artikelid'] == artId]['Namn']