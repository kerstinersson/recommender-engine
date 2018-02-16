# calculates the similarity between two types of wines

# build lookup table

["aglianico"], ["albarino"], ["barbera"], ["cabernetfranc"], ["cabernetsauvignon"], ["carmenere"], ["chardonnay"], ["chenin blanc"], ["corvina"], ["furmint"], ["gamay"], ["garganega"], ["gewurztraminer"], ["godello"], ["grenache"], ["gruner veltliner"], ["malbec"], ["marsanne"], ["melon de bourgogne"], ["merlot"], ["mourvedre"], ["muskat"], ["nebbiolo"], ["negroamaro"], ["nero davola"], ["pinotblanc"], ["pinotage"], ["primitivo"], ["riesling"], ["sangiovese"], ["sauvignon blanc"], ["savagnin"], ["semillon"], ["syrah"], ["shiraz"],  ["tannat"], ["tempranillo"], ["torrontes"], ["touriga nacional"], ["verdicchio"], ["vermentino"], ["viognier"], ["zinfandel"]

lookup = {
	"aglianico": [["aglianico", 4/4], ["albarino", 0/4], ["barbera", 1/4], ["cabernetfranc", 1/4], ["cabernetsauvignon", 1/4], ["carmenere", 1/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 1/4], ["furmint", 0/4], ["gamay", 1/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 1/4], ["gruner veltliner", 0/4], ["malbec", 1/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 1/4], ["mourvedre", 1/4], ["muskat", 0/4], ["nebbiolo", 2/4], ["negroamaro", 2/4], ["nero davola", 1/4], ["pinotblanc", 0/4], ["pinotage", 2/4], ["primitivo", 1/4], ["riesling", 0/4], ["sangiovese", 1/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 1/4], ["shiraz", 1/4],  ["tannat", 2/4], ["tempranillo", 1/4], ["torrontes", 0/4], ["touriga nacional", 1/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 1/4]]
	"albarino": [["aglianico", 0/3], ["albarino", 3/3], ["barbera", 0/3], ["cabernetfranc", 0/3], ["cabernetsauvignon", 0/3], ["carmenere", 0/3], ["chardonnay", 2/3], ["chenin blanc", 1/3], ["corvina", 0/3], ["furmint", 2/3], ["gamay", 0/3], ["garganega", 2/3], ["gewurztraminer", 1/3], ["godello", 2/3], ["grenache", 0/3], ["gruner veltliner", 3/3], ["malbec", 0/3], ["marsanne", 2/3], ["melon de bourgogne", 3/3], ["merlot", 0/3], ["mourvedre", 0/3], ["muskat", 1/3], ["nebbiolo", 0/3], ["negroamaro", 0/3], ["nero davola", 0/3], ["pinotblanc", 2/3], ["pinotage", 0/3], ["primitivo", 0/3], ["riesling", 1/3], ["sangiovese", 0/3], ["sauvignon blanc", 2/3], ["savagnin", 2/3], ["semillon", 2/3], ["syrah", 0/3], ["shiraz", 0/3],  ["tannat", 0/3], ["tempranillo", 0/3], ["torrontes", 2/3], ["touriga nacional", 0/3], ["verdicchio", 2/3], ["vermentino", 2/3], ["viognier", 2/3], ["zinfandel", 0/3]]
	"barbera": [["aglianico", 1/4], ["albarino", 0/4], ["barbera", 4/4], ["cabernetfranc", 2/4], ["cabernetsauvignon", 2/4], ["carmenere", 4/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 2/4], ["furmint", 0/4], ["gamay", 2/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 4/4], ["gruner veltliner", 0/4], ["malbec", 2/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 2/4], ["mourvedre", 2/4], ["muskat", 0/4], ["nebbiolo", 1/4], ["negroamaro", 1/4], ["nero davola", 2/4], ["pinotblanc", 0/4], ["pinotage", 1/4], ["primitivo", 4/4], ["riesling", 0/4], ["sangiovese", 2/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 2/4], ["shiraz", 2/4], ["tannat", 1/4], ["tempranillo", 2/4], ["torrontes", 0/4], ["touriga nacional", 2/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 4/4]]
	"cabernetfranc": [["aglianico", 1/4], ["albarino", 0/4], ["barbera", 2/4], ["cabernetfranc", 4/4], ["cabernetsauvignon", 3/4], ["carmenere", 2/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 3/4], ["furmint", 0/4], ["gamay", 2/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 2/4], ["gruner veltliner", 0/4], ["malbec", 2/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 3/4], ["mourvedre", 2/4], ["muskat", 0/4], ["nebbiolo", 1/4], ["negroamaro", 1/4], ["nero davola", 1/4], ["pinotblanc", 0/4], ["pinotage", 1/4], ["primitivo", 2/4], ["riesling", 0/4], ["sangiovese", 4/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 2/4], ["shiraz", 2/4],  ["tannat", 1/4], ["tempranillo", 3/4], ["torrontes", 0/4], ["touriga nacional", 2/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 2/4]]
	"cabernetsauvignon": [["aglianico", 1/4], ["albarino", 0/4], ["barbera", 2/4], ["cabernetfranc", 3/4], ["cabernetsauvignon", 4/4], ["carmenere", 2/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 3/4], ["furmint", 0/4], ["gamay", 2/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 2/4], ["gruner veltliner", 0/4], ["malbec", 2/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 3/4], ["mourvedre", 2/4], ["muskat", 0/4], ["nebbiolo", 1/4], ["negroamaro", 1/4], ["nero davola", 1/4], ["pinotblanc", 0/4], ["pinotage", 1/4], ["primitivo", 2/4], ["riesling", 0/4], ["sangiovese", 3/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 2/4], ["shiraz", 2/4],  ["tannat", 1/4], ["tempranillo", 4/4], ["torrontes", 0/4], ["touriga nacional", 2/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 2/4]]
	"carmenere": [["aglianico", 1/4], ["albarino", 0/4], ["barbera", 4/4], ["cabernetfranc", 2/4], ["cabernetsauvignon", 2/4], ["carmenere", 4/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 2/4], ["furmint", 0/4], ["gamay", 2/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 4/4], ["gruner veltliner", 0/4], ["malbec", 2/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 2/4], ["mourvedre", 2/4], ["muskat", 0/4], ["nebbiolo", 1/4], ["negroamaro", 1/4], ["nero davola", 1/4], ["pinotblanc", 0/4], ["pinotage", 1/4], ["primitivo", 4/4], ["riesling", 0/4], ["sangiovese", 2/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 2/4], ["shiraz", 2/4],  ["tannat", 1/4], ["tempranillo", 2/4], ["torrontes", 0/4], ["touriga nacional", 2/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 4/4]]
	"chardonnay": [["aglianico", 0/4], ["albarino", 2/3], ["barbera", 0/3], ["cabernetfranc", 0/3], ["cabernetsauvignon", 0/3], ["carmenere", 0/3], ["chardonnay", 3/3], ["chenin blanc", 1/3], ["corvina", 0/3], ["furmint", 2/3], ["gamay", 0/3], ["garganega", 3/3], ["gewurztraminer", 1/3], ["godello", 3/3], ["grenache", 0/3], ["gruner veltliner", 2/3], ["malbec", 0/3], ["marsanne", 2/3], ["melon de bourgogne", 2/3], ["merlot", 0/3], ["mourvedre", 0/3], ["muskat", 1/3], ["nebbiolo", 0/3], ["negroamaro", 0/3], ["nero davola", 0/3], ["pinotblanc", 2/3], ["pinotage", 0/3], ["primitivo", 0/3], ["riesling", 1/3], ["sangiovese", 0/3], ["sauvignon blanc", 2/3], ["savagnin", 3/3], ["semillon", 2/3], ["syrah", 0/3], ["shiraz", 0/3],  ["tannat", 0/3], ["tempranillo", 0/3], ["torrontes", 2/3], ["touriga nacional", 0/3], ["verdicchio", 2/3], ["vermentino", 2/3], ["viognier", 2/3], ["zinfandel", 0/3]]
	"chenin blanc": [["aglianico", 0/3], ["albarino", 1/3], ["barbera", 0/3], ["cabernetfranc", 0/3], ["cabernetsauvignon", 0/3], ["carmenere", 0/3], ["chardonnay", 1/3], ["chenin blanc", 3/3], ["corvina", 0/3], ["furmint", 1/3], ["gamay", 0/3], ["garganega", 1/3], ["gewurztraminer", 3/3], ["godello", 1/3], ["grenache", 0/3], ["gruner veltliner", 1/3], ["malbec", 0/3], ["marsanne", 1/3], ["melon de bourgogne", 1/3], ["merlot", 0/3], ["mourvedre", 0/3], ["muskat", 2/3], ["nebbiolo", 0/3], ["negroamaro", 0/3], ["nero davola", 0/3], ["pinotblanc", 1/3], ["pinotage", 0/3], ["primitivo", 0/3], ["riesling", 3/3], ["sangiovese", 0/3], ["sauvignon blanc", 1/3], ["savagnin", 1/3], ["semillon", 1/3], ["syrah", 0/3], ["shiraz", 0/3],  ["tannat", 0/3], ["tempranillo", 0/3], ["torrontes", 1/3], ["touriga nacional", 0/3], ["verdicchio", 1/3], ["vermentino", 1/3], ["viognier", 1/3], ["zinfandel", 0/3]]
	"corvina": [["aglianico", 1/4], ["albarino", 0/4], ["barbera", 2/4], ["cabernetfranc", 3/4], ["cabernetsauvignon", 3/4], ["carmenere", 2/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 4/4], ["furmint", 0/4], ["gamay", 2/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 2/4], ["gruner veltliner", 0/4], ["malbec", 2/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 2/4], ["mourvedre", 2/4], ["muskat", 0/4], ["nebbiolo", 1/4], ["negroamaro", 1/4], ["nero davola", 1/4], ["pinotblanc", 0/4], ["pinotage", 1/4], ["primitivo", 2/4], ["riesling", 0/4], ["sangiovese", 3/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 2/4], ["shiraz", 2/4],  ["tannat", 1/4], ["tempranillo", 3/4], ["torrontes", 0/4], ["touriga nacional", 2/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 2/4]]
	"furmint": [["aglianico", 0/4], ["albarino", 2/3], ["barbera", 0/3], ["cabernetfranc", 0/3], ["cabernetsauvignon", 0/3], ["carmenere", 0/3], ["chardonnay", 2/3], ["chenin blanc", 1/3], ["corvina", 0/3], ["furmint", 3/3], ["gamay", 0/3], ["garganega", 2/3], ["gewurztraminer", 1/3], ["godello", 2/3], ["grenache", 0/3], ["gruner veltliner", 2/3], ["malbec", 0/3], ["marsanne", 3/3], ["melon de bourgogne", 2/3], ["merlot", 0/3], ["mourvedre", 0/3], ["muskat", 1/3], ["nebbiolo", 0/3], ["negroamaro", 0/3], ["nero davola", 0/3], ["pinotblanc", 2/3], ["pinotage", 0/3], ["primitivo", 0/3], ["riesling", 1/3], ["sangiovese", 0/3], ["sauvignon blanc", 2/3], ["savagnin", 2/3], ["semillon", 1/3], ["syrah", 0/3], ["shiraz", 0/3],  ["tannat", 0/3], ["tempranillo", 0/3], ["torrontes", 3/3], ["touriga nacional", 0/3], ["verdicchio", 2/3], ["vermentino", 2/3], ["viognier", 3/3], ["zinfandel", 0/3]]
	"gamay": [["aglianico", 1/4], ["albarino", 0/4], ["barbera", 2/4], ["cabernetfranc", 2/4], ["cabernetsauvignon", 2/4], ["carmenere", 2/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 2/4], ["furmint", 0/4], ["gamay", 4/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 2/4], ["gruner veltliner", 0/4], ["malbec", 2/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 2/4], ["mourvedre", 2/4], ["muskat", 0/4], ["nebbiolo", 1/4], ["negroamaro", 1/4], ["nero davola", 1/4], ["pinotblanc", 0/4], ["pinotage", 1/4], ["primitivo", 2/4], ["riesling", 0/4], ["sangiovese", 2/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 2/4], ["shiraz", 2/4],  ["tannat", 1/4], ["tempranillo", 2/4], ["torrontes", 0/4], ["touriga nacional", 2/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 2/4]]
	"garganega": [["aglianico", 0/3], ["albarino", 2/3], ["barbera", 0/3], ["cabernetfranc", 0/3], ["cabernetsauvignon", 0/3], ["carmenere", 0/3], ["chardonnay", 3/3], ["chenin blanc", 1/3], ["corvina", 0/3], ["furmint", 2/3], ["gamay", 0/3], ["garganega", 3/3], ["gewurztraminer", 1/3], ["godello", 3/3], ["grenache", 0/3], ["gruner veltliner", 2/3], ["malbec", 0/3], ["marsanne", 2/3], ["melon de bourgogne", 1/3], ["merlot", 0/3], ["mourvedre", 0/3], ["muskat", 1/3], ["nebbiolo", 0/3], ["negroamaro", 0/3], ["nero davola", 0/3], ["pinotblanc", 2/3], ["pinotage", 0/3], ["primitivo", 0/3], ["riesling", 1/3], ["sangiovese", 0/3], ["sauvignon blanc", 2/3], ["savagnin", 3/3], ["semillon", 1/3], ["syrah", 0/3], ["shiraz", 0/3],  ["tannat", 0/3], ["tempranillo", 0/3], ["torrontes", 2/3], ["touriga nacional", 0/3], ["verdicchio", 2/3], ["vermentino", 2/3], ["viognier", 3/3], ["zinfandel", 0/3]]
	"gewurztraminer": [["aglianico", 0/3], ["albarino", 1/3], ["barbera", 0/3], ["cabernetfranc", 0/3], ["cabernetsauvignon", 0/3], ["carmenere", 0/3], ["chardonnay", 1/3], ["chenin blanc", 3/3], ["corvina", 0/3], ["furmint", 1/3], ["gamay", 0/3], ["garganega", 1/3], ["gewurztraminer", 3/3], ["godello", 1/3], ["grenache", 0/3], ["gruner veltliner", 1/3], ["malbec", 0/3], ["marsanne", 1/3], ["melon de bourgogne", 2/3], ["merlot", 0/3], ["mourvedre", 0/3], ["muskat", 2/3], ["nebbiolo", 0/3], ["negroamaro", 0/3], ["nero davola", 0/3], ["pinotblanc", 1/3], ["pinotage", 0/3], ["primitivo", 0/3], ["riesling", 3/3], ["sangiovese", 0/3], ["sauvignon blanc", 1/3], ["savagnin", 1/3], ["semillon", 2/3], ["syrah", 0/3], ["shiraz", 0/3], ["tannat", 0/3], ["tempranillo", 0/3], ["torrontes", 1/3], ["touriga nacional", 0/3], ["verdicchio", 1/3], ["vermentino", 1/3], ["viognier", 1/3], ["zinfandel", 0/3]]
	"godello": [["aglianico", 0/3], ["albarino", 2/3], ["barbera", 0/3], ["cabernetfranc", 2/3], ["cabernetsauvignon", 2/3], ["carmenere", 2/3], ["chardonnay", 3/3], ["chenin blanc", 1/3], ["corvina", 0/3], ["furmint", 3/3], ["gamay", 0/3], ["garganega", 3/3], ["gewurztraminer", 1/3], ["godello", 3/3], ["grenache", 0/3], ["gruner veltliner", 2/3], ["malbec", 0/3], ["marsanne", 2/3], ["melon de bourgogne", 1/3], ["merlot", 0/3], ["mourvedre", 0/3], ["muskat", 1/3], ["nebbiolo", 0/3], ["negroamaro", 0/3], ["nero davola", 0/3], ["pinotblanc", 2/3], ["pinotage", 0/3], ["primitivo", 0/3], ["riesling", 1/3], ["sangiovese", 0/3], ["sauvignon blanc", 2/3], ["savagnin", 3/3], ["semillon", 1/3], ["syrah", 0/3], ["shiraz", 0/3], ["tannat", 0/3], ["tempranillo", 0/3], ["torrontes", 2/3], ["touriga nacional", 0/3], ["verdicchio", 2/3], ["vermentino", 2/3], ["viognier", 3/3], ["zinfandel", 0/3]]
	"grenache": [["aglianico", 1/4], ["albarino", 0/4], ["barbera", 4/4], ["cabernetfranc", 2/4], ["cabernetsauvignon", 2/4], ["carmenere", 4/4], ["chardonnay", 0/4], ["chenin blanc", 0/4], ["corvina", 2/4], ["furmint", 0/4], ["gamay", 2/4], ["garganega", 0/4], ["gewurztraminer", 0/4], ["godello", 0/4], ["grenache", 4/4], ["gruner veltliner", 0/4], ["malbec", 2/4], ["marsanne", 0/4], ["melon de bourgogne", 0/4], ["merlot", 2/4], ["mourvedre", 2/4], ["muskat", 0/4], ["nebbiolo", 1/4], ["negroamaro", 1/4], ["nero davola", 1/4], ["pinotblanc", 0/4], ["pinotage", 1/4], ["primitivo", 4/4], ["riesling", 0/4], ["sangiovese", 2/4], ["sauvignon blanc", 0/4], ["savagnin", 0/4], ["semillon", 0/4], ["syrah", 2/4], ["shiraz", 2/4],  ["tannat", 1/4], ["tempranillo", 2/4], ["torrontes", 0/4], ["touriga nacional", 2/4], ["verdicchio", 0/4], ["vermentino", 0/4], ["viognier", 0/4], ["zinfandel", 4/4]]
	"gruner veltliner": [["aglianico", 0/3], ["albarino", 3/3], ["barbera", 0/3], ["cabernetfranc", 0/3], ["cabernetsauvignon", 0/3], ["carmenere", 0/3], ["chardonnay", 2/3], ["chenin blanc", 1/3], ["corvina", 0/3], ["furmint", 2/3], ["gamay", 0/3], ["garganega", 2/3], ["gewurztraminer", 1/3], ["godello", 2/3], ["grenache", 0/3], ["gruner veltliner", 3/3], ["malbec", 0/3], ["marsanne", 2/3], ["melon de bourgogne", 1/3], ["merlot", 0/3], ["mourvedre", 0/3], ["muskat", 1/3], ["nebbiolo", 0/3], ["negroamaro", 0/3], ["nero davola", 0/3], ["pinotblanc", 2/3], ["pinotage", 0/3], ["primitivo", 0/3], ["riesling", 1/3], ["sangiovese", 0/3], ["sauvignon blanc", 2/3], ["savagnin", 2/3], ["semillon", 1/3], ["syrah", 0/3], ["shiraz", 0/3],  ["tannat", 0/3], ["tempranillo", 0/3], ["torrontes", 2/3], ["touriga nacional", 0/3], ["verdicchio", 2/3], ["vermentino", 2/3], ["viognier", 2/3], ["zinfandel", 0/3]]
	"malbec": [["aglianico"], ["albarino"], ["barbera"], ["cabernetfranc"], ["cabernetsauvignon"], ["carmenere"], ["chardonnay"], ["chenin blanc"], ["corvina"], ["furmint"], ["gamay"], ["garganega"], ["gewurztraminer"], ["godello"], ["grenache"], ["gruner veltliner"], ["malbec"], ["marsanne"], ["melon de bourgogne"], ["merlot"], ["mourvedre"], ["muskat"], ["nebbiolo"], ["negroamaro"], ["nero davola"], ["pinotblanc"], ["pinotage"], ["primitivo"], ["riesling"], ["sangiovese"], ["sauvignon blanc"], ["savagnin"], ["semillon"], ["syrah"], ["shiraz"],  ["tannat"], ["tempranillo"], ["torrontes"], ["touriga nacional"], ["verdicchio"], ["vermentino"], ["viognier"], ["zinfandel"]]
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