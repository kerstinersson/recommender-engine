# calculates the similarity between two types of wines

# build lookup table

["aglianico"], ["albarino"], ["barbera"], ["cabernetfranc"], ["cabernetsauvignon"], ["carmenere"], ["chardonnay"], ["chenin blanc"], ["corvina"], ["furmint"], ["gamay"], ["garganega"], ["gewurztraminer"], ["godello"], ["grenache"], ["gruner veltliner"], ["malbec"], ["marsanne"], ["melon de bourgogne"], ["merlot"], ["mourvedre"], ["muskat"], ["nebbiolo"], ["negroamaro"], ["nero davola"], ["pinotblanc"], ["pinotage"], ["primitivo"], ["riesling"], ["sangiovese"], ["sauvignon blanc"], ["savagnin"], ["semillon"], ["syrah"], ["shiraz"],  ["tannat"], ["tempranillo"], ["torrontes"], ["touriga nacional"], ["verdicchio"], ["vermentino"], ["viognier"], ["zinfandel"]

lookup = {
	"aglianico": [["aglianico", 4/4], ["albarino", 0/4], ["barbera", 1/4], ["cabernetfranc", 1/4], ["cabernetsauvignon", 1/4], ["carmenere", 1/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 1/4], ["furmint", 0/4], ["gamay", 1/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 1/4], ["gruner veltliner", 0/4], ["malbec", 1/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 1/4], ["mourvedre", 1/4], ["muskat", 0/4], ["nebbiolo", 2/4], ["negroamaro", 2/4], ["nero davola", 1/4], ["pinotblanc", 0/4], ["pinotage", 2/4], ["primitivo", 1/4], ["riesling", 0/4], ["sangiovese", 1/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 1/4], ["shiraz", 1/4],  ["tannat", 2/4], ["tempranillo", 1/4], ["torrontes", 0/4], ["touriga nacional", 1/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 1/4]]
	"albarino": [["aglianico", 0/3], ["albarino", 3/3], ["barbera", 0/3], ["cabernetfranc", 0/3], ["cabernetsauvignon", 0/3], ["carmenere", 0/3], ["chardonnay", 2/3], ["chenin blanc", 1/3], ["corvina", 0/3], ["furmint", 2/3], ["gamay", 0/3], ["garganega", 2/3], ["gewurztraminer", 1/3], ["godello", 2/3], ["grenache", 0/3], ["gruner veltliner", 3/3], ["malbec", 0/3], ["marsanne", 2/3], ["melon de bourgogne", 3/3], ["merlot", 0/3], ["mourvedre", 0/3], ["muskat", 1/3], ["nebbiolo", 0/3], ["negroamaro", 0/3], ["nero davola", 0/3], ["pinotblanc", 2/3], ["pinotage", 0/3], ["primitivo", 0/3], ["riesling", 1/3], ["sangiovese", 0/3], ["sauvignon blanc", 2/3], ["savagnin", 2/3], ["semillon", 2/3], ["syrah", 0/3], ["shiraz", 0/3],  ["tannat", 0/3], ["tempranillo", 0/3], ["torrontes", 2/3], ["touriga nacional", 0/3], ["verdicchio", 2/3], ["vermentino", 2/3], ["viognier", 2/3], ["zinfandel", 0/3]]
	"barbera": [["aglianico", 1/4], ["albarino", 0/4], ["barbera", 4/4], ["cabernetfranc", 2/4], ["cabernetsauvignon", 2/4], ["carmenere", 4/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 2/4], ["furmint", 0/4], ["gamay", 2/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 4/4], ["gruner veltliner", 0/4], ["malbec", 2/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 2/4], ["mourvedre", 2/4], ["muskat", 0/4], ["nebbiolo", 1/4], ["negroamaro", 1/4], ["nero davola", 2/4], ["pinotblanc", 0/4], ["pinotage", 1/4], ["primitivo", 4/4], ["riesling", 0/4], ["sangiovese", 2/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 2/4], ["shiraz", 2/4], ["tannat", 1/4], ["tempranillo", 2/4], ["torrontes", 0/4], ["touriga nacional", 2/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 4/4]]
	"cabernetfranc": {}
	"cabernetsauvignon": {}
	"carmenere": {}
	"chardonnay": {}
	"chenin blanc": {}
	"corvina": {}
	"furmint": {}
	"gamay": {}
	"garganega": {}
	"gewurztraminer": {}
	"godello": {}
	"grenache": {}
	"gruner veltliner": {}
	"malbec": {}
	"marsanne": {}
	"melon de bourgogne": {}
	"merlot": {}
	"mourvedre": {}
	"muskat": {}
	"nebbiolo": {}
	"negroamaro": {}
	"nero davola": {}
	"pinotblanc": {}
	"pinotgris": {}
	"pinotnoir": {}
	"pinotage": {}
	"primitivo": {}
	"riesling": {}
	"sangiovese": {}
	"sauvignon blanc": {}
	"savagnin": {}
	"semillon": {}
	"syrah": {}
	"shiraz": {}
	"tannat": {}
	"tempranillo": {}
	"torrontes": {}
	"touriga nacional": {}
	"verdicchio": {}
	"vermentino": {}
	"viognier": {}
	"zinfandel": {}
}


def sim_wines(w1, w2):
	if w1 is None or w2 is None:
		print("No type of wine given...")
		return 0
	else:
		print("calculating similarity")
		return 1

if __name__ == '__main__':
	sim_wines(None, "Wow")