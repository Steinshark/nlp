import string 
import re 



QUOTATIONS      = {
    '‘': "'", '’': "'", '‚': ",", '‛': "'",
    '“': '"', '”': '"', '„': ',', '‟': '"',
    '′': "'", '″': '"'
}

DASHES          = {
    '–': '-', '—': '-', '―': '-', '−': '-'
}

SPACING         = {
    # Ellipsis
        '…': '...',
        
        # Spaces
        '\u00a0': ' ',  # non-breaking space
        '\u200b': '',   # zero-width space
        '\u200c': '',   # zero-width non-joiner
        '\u200d': '',   # zero-width joiner
        '\u202f': ' ',  # narrow no-break space
        '\u2060': '',   # word joiner
}


CHAR_CORRECTIONS = {
        " i i ": " i ",        'ç':"c",
        'с':"c",        'π':"pi",
        '≅':"~=",         '™':"TM",
        '𝑛':"n",          '𝑃':"P", 
        'ℙ':"P",         '−':"-", 
        '²':"^2",         '𝜋':"pi", 
        '𝐸':"E",         '~':"~", 
        'γ':"gamma",         '′':"`", 
        '¹':"^1",         '⁵':"^5", 
        'в':"B",         '𝐺':"G", 
        '₂':"_2",         '∀':" for all ", 
        'м':"M",        '∃':" there exists ",
        'Δ':"Delta",         '𝜃':"Theta", 
        '‽':"?!",         '𝛿':"sigma", 
        'ő':"o",         '𝐻':"H", 
        '�':"?",        '∈':" element of ",
        '₁':"_1",        'δ':"sigma", 
        '∎':"[]",         '⊗':"X", 
        'ɸ':"phi",         'ν':"v", 
        'ℕ':"N",         '\u2009':"?",
        '𝐷':"D",         '·':" dot ",
        'ä':"a",        '̶':"?", 
        '⁰':"degrees",         'É':"E",
        'à':"a",        'е':"e",
        'д':"D",        '×':"x",
        '→':"->",         'ö':"o",
        'Ο':"O",        '𝐶':"C",
        '𝑎':"alpha",        'ú':"u",
        'т':"T",        '𝐹':"F",
        '½':"1/2",        'ℝ':"R", 
        'θ':"Theta",         'έ':"e", 
        'ô':"o",         '³':"^3", 
        'á':'a',         '𝓁':"l", 
        '´':"`",         'ń':"n", 
        '⅓':"1/3",         'ï':"l", 
        '･':"dot",         '–':"-", 
        '𝐵':"B",
        '𝑏':"b",         'č':"c",
        '∂':"b",         '¡':"!", 
        'ü':"u",         '⁴':"^4", 
        'ᵢ':"_i",       'Ö':"O",
        'υ':"v",         '😲':":)",
        'ë':"e",        'ã':"a",
        'é':"e",        'á':'a',
        'ö':'o',        'â':'a',
        'ā':"a",        'š':"s",
        'ř':"r",        "ō":"o",
        "õ":"o",        'й':"n",
        'ì':"i",        'ī':"i",
        'Š':"S",        'ù':"u",
       
        '𝑧':"z", '⅔':"2/3", '¼':"1/4", 'ω':"w", '𝑤':"w",
        # Quotes
        '‘': "'", '’': "'", '‚': ",", '‛': "'",
        '“': '"', '”': '"', '„': ',', '‟': '"',
        
        # Dashes and hyphens
        '–': '-', '—': '-', '―': '-', '−': '-',  # minus sign too
        
        # Ellipsis
        '…': '...',
        
        # Spaces
        '\u00a0': ' ',  # non-breaking space
        '\u200b': '',   # zero-width space
        '\u200c': '',   # zero-width non-joiner
        '\u200d': '',   # zero-width joiner
        '\u202f': ' ',  # narrow no-break space
        '\u2060': '',   # word joiner

        # Misc
        '•': '*', '·': '*', '§': 'S', '°': ' deg ',
        '‹': '<', '›': '>', '«': '<<', '»': '>>',
        '′': "'", '″': '"',
        '†': '+', '‡': '++', '‒': '-', '※': '*',
        '¤': '$', '€': 'EUR', '£': 'GBP', '¥': 'JPY',
        '©': '(c)', '®': '(r)', '™': '(tm)',
}

COMMON_MISSPELLINGS = {
    "teh": "the",
    "recieve": "receive",
    "definately": "definitely",
    "occured": "occurred",
    "seperate": "separate",
    "untill": "until",
    "wich": "which",
    "beggining": "beginning",
    "concious": "conscious",
    "adress": "address",
    "tommorow": "tomorrow",
    "enviroment": "environment",
    "thier": "their",
    "goverment": "government",
    "occurence": "occurrence",
    "acommodate": "accommodate",
    "arguement": "argument",
    "embarass": "embarrass",
    "independant": "independent",
    "neccessary": "necessary",
    "publically": "publicly",
    "restarant": "restaurant",
    "succesful": "successful",
    "trully": "truly",
    "wierd": "weird",
    "labled": "labeled",
    "calender": "calendar",
    "embarased": "embarrassed",
    "occuring": "occurring",
    "dissapear": "disappear",
    "manuever": "maneuver",
    "commited": "committed",
    "neccessarily": "necessarily",
}

TECH_MISSPELLINGS = {
    "algorithim": "algorithm",
    "pyhton": "python",
    "jvascript": "javascript",
    "javscript": "javascript",
    "javascritp": "javascript",
    "htlm": "html",
    "csss": "css",
    "caml": "camel",
    "funciton": "function",
    "funtion": "function",
    "intial": "initial",
    "inital": "initial",
    "lenght": "length",
    "widht": "width",
    "paramter": "parameter",
    "parmeter": "parameter",
    "instace": "instance",
    "modle": "model",
    "netowrk": "network",
    "pakage": "package",
    "packge": "package",
    "repositry": "repository",
    "reposotory": "repository",
    "dependancy": "dependency",
    "dependecy": "dependency",
    "depedency": "dependency",
    "intrepreter": "interpreter",
    "framwork": "framework",
    "framwrok": "framework",
    "libary": "library",
    "librery": "library",
    "datbase": "database",
    "databse": "database",
    "databsae": "database",
    "statment": "statement",
    "statemnt": "statement",
    "retun": "return",
    "retrun": "return",
    "retrive": "retrieve",
    "pritn": "print",
    "rturn": "return",
    "evalute": "evaluate",
    "syntx": "syntax",
    "intput": "input",
    "ouput": "output",
    "booolean": "boolean",
    "encapsulationg": "encapsulation",
    "inhertance": "inheritance",
    "polymorphysm": "polymorphism",
    "referance": "reference",
    "documnet": "document",
    "statstics": "statistics",
    "heurisitic": "heuristic",
    "tokeniztion": "tokenization",
    "vectorzation": "vectorization",
    "embeding": "embedding",
    "traning": "training",
    "tranformers": "transformers",
}

SPACING_MISSPELLINGS = {
    "machinelearning": "machine learning",
    "deeplearning": "deep learning",
    "neuralnetworks": "neural networks",
    "supportvectormachine": "support vector machine",
    "recurrentneuralnet": "recurrent neural net",
    "naturalanguage": "natural language",
}

BRITISH_MISSPELLINGS = {
    "colour": "color",
    "favour": "favor",
    "honour": "honor",
    "labour": "labor",
    "neighbour": "neighbor",
    "organise": "organize",
    "organised": "organized",
    "organising": "organizing",
    "realise": "realize",
    "realised": "realized",
    "realising": "realizing",
    "recognise": "recognize",
    "recognised": "recognized",
    "recognising": "recognizing",
    "analyse": "analyze",
    "analysed": "analyzed",
    "analysing": "analyzing",
    "defence": "defense",
    "licence": "license",
    "offence": "offense",
    "pretence": "pretense",
    "centre": "center",
    "metre": "meter",
    "litre": "liter",
    "theatre": "theater",
    "travelling": "traveling",
    "travelled": "traveled",
    "traveller": "traveler",
    "modelling": "modeling",
    "modelled": "modeled",
    "cancelling": "canceling",
    "jewellery": "jewelry",
    "storey": "story",
    "tyre": "tire",
    "aluminium": "aluminum",
    "aeroplane": "airplane",
    "cheque": "check",
    "plough": "plow",
    "programme": "program",
    "catalogue": "catalog",
    "dialogue": "dialog",
    "grey": "gray",
    "mould": "mold ",
    "moult": "molt",
    "pyjamas": "pajamas",
    "sceptical": "skeptical",
    "enrol": "enroll"
    }

# Combine and space-pad each entry
ALL_MISSPELLINGS = {}
for d in [COMMON_MISSPELLINGS, TECH_MISSPELLINGS, SPACING_MISSPELLINGS, BRITISH_MISSPELLINGS]:
    for k, v in d.items():
        ALL_MISSPELLINGS[f" {k} "] = f" {v} "


ALL_CORRECTIONS     = {}
for d in [CHAR_CORRECTIONS, ALL_MISSPELLINGS]:
    for k, v in d.items():
        ALL_CORRECTIONS[f"{k}"] = f"{v}"


PATTERN = re.compile('|'.join(re.escape(k) for k in ALL_CORRECTIONS.keys()))

REMOVAL_THRESH          = 2e-7
REMOVAL_CHAR            = ''
EOT_STR                 = '<|endoftext|>'
GOOD_CHAR               = string.ascii_letters + "1234567890" + "!@#$%^&*()_+-={}[]:;',./<>?~`|\\" + '"'

ONLYASCII               = re.compile('|'.join(re.escape(c) for c in GOOD_CHAR))

def match_case(word: str, template: str) -> str:
    if template.isupper():
        return word.upper()
    elif template[0].isupper():
        return word.capitalize()
    else:
        return word.lower()
    
#Create regex pattern using quotations, dashes, spacing
boundary_corrections             = {}  
for correction_set in [COMMON_MISSPELLINGS,BRITISH_MISSPELLINGS,SPACING_MISSPELLINGS]:
    boundary_corrections.update(correction_set)

compiled_boundary_corrections    = [(re.compile(rf"\b{re.escape(wrong)}\b", flags=re.IGNORECASE), right) for wrong, right in boundary_corrections.items()]


nonboundary_corrections             = {}  
for correction_set in [QUOTATIONS,DASHES,SPACING]:
    nonboundary_corrections.update(correction_set)

compiled_nonboundary_corrections    = [(re.compile(rf"{re.escape(wrong)}", flags=re.IGNORECASE), right) for wrong, right in nonboundary_corrections.items()]


"""
This function performs staple normalizations to include: 
    - format all special punctuation to a standard format (we dont need 3 types of dashes)
"""
def normalize_text(text:str):
    
    #Perform word and utf-8 level corrections
    for pattern, replacement in compiled_boundary_corrections:
        def repl(match):
            return match_case(replacement, match.group(0))
        text = pattern.sub(repl, text)

    for pattern, replacement in compiled_nonboundary_corrections:
        text = pattern.sub(replacement, text)

    return text 


if __name__ =='__main__':
    text = "“hello— -world”. Teh world will be Jewellery."
    print(f"before: '{text}'")
    print(normalize_text(text))