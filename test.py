# functions to clean beskrivningravaror feature in data

str1 = "Chardonnay and pinot noir och cabernet franc."
str2 = "50%% chenin blanc, 35%% chardonnay och 15%% sauvignon blanc."

# remove percentages
str2 = filter(lambda x: x.isalpha() or x == " ", str2)

# write cabernets and pinots as one word to avoid confusion
repls = ("pinot ", "pinot"), ("cabernet ", "cabernet")
str1 = reduce(lambda a, kv: a.replace(*kv), repls, str1)