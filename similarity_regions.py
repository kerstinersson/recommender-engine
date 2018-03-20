# calculates the similarity between regions
reg = ['venetien' 'rioja' '' 'apulien' 'champagne' 'cava' 'lisboa' 'pfalz' 'kalifornien' 'piemonte' 'bourgogne' 'bordeaux' 'loiredalen' 'castillayle\xc3\xb3n' 'istra' 'westerncape' 'marlborough' 'wellingtonnz' 'rhonedalen' 'southaustralia' 'southeasternaustralia' 'cuyo' 'vallecentral' 'westernaustralia' 'aconcagua' 'tokaj-hegyalja' 'alicante' 'mosel' 'toscana' 'douro' 'regiondelsur' 'morava' 'languedoc-roussillon' 'rheingau' 'england' 'alsace' 'washingtonstate' 'niederosterreich' 'priorat' 'lombardiet' 'trakien' 'vdltcastilla' 'sardinien' 'valencia' 'sicilien' 'marche' 'savoie' 'rheinhessen' 'emilia-romagna' 'kampanien' 'serraga\xc3\xbacha' 'gotlandslan' 'frankrikesydvast' 'tejo' 'dobrogea' 'danubeplain' 'terraalta' 'skanelan' 'ribeirasacra' 'riberadelduero' 'catalunya' 'canterbury' 'beira' 'montsant' 'abruzzerna' 'pened\xc3\xa8s' 'bierzo' 'cari\xc3\xb1ena' 'baden' 'kalmarlan' 'victoria' 'kakhetiregion' 'manchuela' 'valdeorras' 'peloponnesos' "hawke'sbay" 'lamancha' 'burgenland' 'newsouthwales' 'salta' 'costersdelsegre' 'rapel' 'oregon' 'trentino-altoadige' 'eger' 'terrasdosado' 'vinospumantediqualit\xc3\xa0deltipoaromatico' 'valdepe\xc3\xb1as' 'korsika' 'maipo' 'maule' 'mediterranee' 'bekaa' 'attica' 'toledo' 'ribeiro' 'jura' 'getariakotxakolina' 'podravina' 'sopron' 'pen\xc3\xadnsuladeset\xc3\xbabal' 'santorini' 'tasmanien' 'znojmo' 'navarra' 'del-balaton' 'rueda' 'primorski' 'minho' 'arlanza' 'valais' 'latium' 'blekingelan' 'newyorkstate' 'alentejo' 'golanhojderna(israeliskbosattning)' 'patagonien' 'kalabrien' 'yecla' 'vinosdemadrid' 'provence' 'r\xc3\xadasbaixas' 'centralotago' 'makedonien' 'toro' 'nagy-soml\xc3\xb3i' 'larioja' 'mosel-saar-ruwer' 'jumilla' 'rhein' 'friuli-venezia-giulia' 'coquimbo' 'molise' 'w\xc3\xbcrttemberg' 'ligurien' 'nahe' 'vdltdemurc\xc3\xada' 'gisborne' 'sodermanlandslan' 'sekt' 'salamanca' 'nelson' 'utiel-requena' 'somontano' 'franken' 'valledelaorotava' 'campodeborja' 'malaga' 'umbrien' 'valedosvinhedos' 'britishcolumbia' 'jamtlandslan' 'povardarje' 'samos' 'kreta' 'duna\xc3\xa2\xe2\x82\xac\xe2\x80\x9ctiszakozi' 'primorskahrvatska']

# Lookup table with all regions and their climate type
regions = {'venetien': '',
	'rioja': 'cool',
	'apulien': 'warm', 
	'champagne': 'cool',
 	'cava': 'warm',
 	'lisboa': 'warm',
 	'pfalz': 'cool',
 	'kalifornien': 'warm',
 	'piemonte': 'intermediate',
 	'bourgogne': 'cool',
 	'bordeaux': 'intermediate',
 	'loiredalen': 'cool',
 	'castillayle\xc3\xb3n': 'intermediate',
 	'istra': 'warm', 
 	'westerncape': 'warm',
  	'marlborough': 'cool',
  	'wellingtonnz': 'cool',
  	'rhonedalen': 'warm',
  	'southaustralia': 'warm',
  	'southeasternaustralia': 'warm',
  	'cuyo': 'warm',
  	'vallecentral': 'intermediate',
  	'westernaustralia': 'warm',
  	'aconcagua': 'warm',
  	'tokaj-hegyalja': 'intermediate',
  	'alicante': 'warm',
  	'mosel': 'cool',
  	'toscana': 'intermediate',
  	'douro': 'warm',
 	'regiondelsur': 'intermediate',
 	'morava': 'cool',
 	'languedoc-roussillon': 'warm',
 	'rheingau': 'cool',
	'england': 'cool',
	'alsace': 'cool',
	'washingtonstate': 'intermediate',
  	'niederosterreich': 'cool',
  	'priorat': 'warm',
  	'lombardiet': 'cool',
  	'trakien': 'warm',
  	'vdltcastilla': 'intermediate',
  	'sardinien': 'warm',
  	'valencia': 'warm',
  	'sicilien': 'warm',
  	'marche': 'intermediate',
  	'savoie': 'warm',
  	'rheinhessen': 'cool',
  	'emilia-romagna': 'intermediate',
  	'kampanien': 'warm',
  	'serraga\xc3\xbacha': 'warm',
  	'gotlandslan': 'cool',
  	'frankrikesydvast': 'intermediate',
  	'tejo': 'warm',
  	'dobrogea': 'intermediate',
  	'danubeplain': 'warm',
  	'terraalta': 'intermediate',
  	'skanelan': 'cool',
  	'ribeirasacra': 'cool',
  	'riberadelduero': 'warm',
  	'catalunya': 'warm',
  	'canterbury': 'cool',
  	'beira': 'intermediate',
  	'montsant': 'warm',
  	'abruzzerna': 'intermediate',
  	'pened\xc3\xa8s': 'warm',
  	'bierzo': 'warm',
  	'cari\xc3\xb1ena': 'warm',
  	'baden': 'cool',
  	'kalmarlan': 'cool',
  	'victoria': 'cool',
  	'kakhetiregion': 'cool',
  	'manchuela': 'intermediate',
  	'valdeorras': 'cool',
  	'peloponnesos': 'warm',
  	"hawke'sbay": 'warm',
  	'lamancha': 'intermediate',
  	'burgenland': 'cool',
  	'newsouthwales': 'warm',
  	'salta': 'warm',
  	'costersdelsegre': 'intermediate',
  	'rapel': 'warm',
  	'oregon': 'intermediate',
  	'trentino-altoadige': 'warm',
  	'eger': 'warm',
  	'terrasdosado': 'warm',
	'vinospumantediqualit\xc3\xa0deltipoaromatico': 'intermediate',
	'valdepe\xc3\xb1as': 'warm',
	'korsika': 'warm',
	'maipo': 'warm',
	'maule': 'warm',
	'mediterranee': 'warm',
	'bekaa': 'warm',
	'attica': 'warm',
	'toledo': 'warm',
	'ribeiro': 'warm',
	'jura': 'cool',
	'getariakotxakolina': 'warm',
	'podravina': 'cool',
	'sopron': 'warm',
	'pen\xc3\xadnsuladeset\xc3\xbabal': 'warm',
	'santorini': 'warm',
	'tasmanien': 'intermediate',
	'znojmo': '',
	'navarra' 'del-balaton' 'rueda' 'primorski' 'minho' 'arlanza' 'valais' 'latium' 'blekingelan' 'newyorkstate' 'alentejo' 'golanhojderna(israeliskbosattning)' 'patagonien' 'kalabrien' 'yecla' 'vinosdemadrid' 'provence' 'r\xc3\xadasbaixas' 'centralotago' 'makedonien' 'toro' 'nagy-soml\xc3\xb3i' 'larioja' 'mosel-saar-ruwer' 'jumilla' 'rhein' 'friuli-venezia-giulia' 'coquimbo' 'molise' 'w\xc3\xbcrttemberg' 'ligurien' 'nahe' 'vdltdemurc\xc3\xada' 'gisborne' 'sodermanlandslan' 'sekt' 'salamanca' 'nelson' 'utiel-requena' 'somontano' 'franken' 'valledelaorotava' 'campodeborja' 'malaga' 'umbrien' 'valedosvinhedos' 'britishcolumbia' 'jamtlandslan' 'povardarje' 'samos' 'kreta' 'duna\xc3\xa2\xe2\x82\xac\xe2\x80\x9ctiszakozi' 'primorskahrvatska'}

def sim_regions(reg1, reg2):
	if reg1 in regions:
		type1 = regions[reg1]

		if reg2 in regions:
			type2 = regions[reg2]

			if type1 == type2:
				return 1
			else if type1 == "intermediate" or type2 == "intermediate"
				return 0.5
			else: 
				return 0
		else:
			return 0
	else: 
		return 0
