# check if grape is synonymous to another

grapes = {
	"agiorgitiko": [],
	"aglianico": [],
	"albarino": [],
	"albillo": ["albilla"],
	"arneis": ["nebbioloblanco", "bianchetto"],
	"barbera": [],
	"blaufrankisch": ["kekfrankos"],
	"bobal": [],
	"bonarda": [],
	"bourboulenc": ["clairette"],
	"brunello": [],
	"cabernetfranc": [],
	"cabernetsauvignon": [],
	"carignan": ["mazuelo"],
	"carmenere": [],
	"catarratto": ["carricante"],
	"chardonnay": [],
	"cheninblanc": [],
	"ciliegiolo": [],
	"cinsaut": ["cinsault"],
	"corvina": ["corvinone", "rondinella", "molinara"],
	"falanghina": [],
	"furmint": [],
	"gamay": [],
	"garganega": [],
	"gewurztraminer": [],
	"glera": [],
	"godello": [],
	"gouveio": [],
	"graciano": [],
	"grenache": ["marselan", "cannonau"],
	"grunerveltliner": [],
	"jacquere": [],
	"macabeo": ["viura", "macabeu"],
	"malbec": [],
	"malvasia": [],
	"marsanne": [],
	"melondebourgogne": [],
	"mencia": [],
	"merlot": [],
	"montepulciano": [],
	"mourvedre": ["monastrell"],
	"muscadet": ["muscadelle"],
	"muller-thurgau": [],
	"muskat": ["zibibbo"],
	"nebbiolo": ["dolcetto"],
	"negroamaro": [],
	"nerodavola": [],
	"parellada": [],
	"petitsirah": ["durif"],
	"pinotblanc": [],
	"pinotgris": ["pinotgrigio", "grauburgunder"],
	"pinotmeunier": [],
	"pinotnoir": ["spatburgunder"],
	"pinotage": [],
	"primitivo": [],
	"rabigado": [],
	"riesling": [],
	"rondo": [],
	"sangiovese": ["canaiolo"],
	"sauvignonblanc": [],
	"savagnin": ["verdejo"],
	"semillon": [],
	"silvaner": [],
	"solaris": [],
	"syrah": [],
	"shiraz": [],
	"tannat": [],
	"tempranillo": ["aragonez", "tintaroriz"],
	"torrontes": [],
	"touriganacional": ["tourigafranca"],
	"verdicchio": [],
	"verdot": ["petit verdot"],
	"verduzzo": [],
	"vermentino": [],
	"vespolina": [],
	"viognier": [],
	"viosigno": [],
	"zinfandel": []
	"xarel-lo": []
}

def check_synonym(grape):
	for grape in grapes.keys():
		if grapes[grape] == grape:
			return grape

	return None