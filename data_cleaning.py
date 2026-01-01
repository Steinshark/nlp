import numpy 
import xml.etree.ElementTree as ET
import multiprocessing
import time 
from training import *
import random 
import tqdm 
import pandas 
import re 
import unicodedata
import language_utils
from datagrab import stream_dataset


hard_bad_words = [
    "viagra", "cunt", "fuck", "slut", "pussy", "slots", "testosterone booster",
    "miracle cure", "weight loss pill", "hair loss treatment", "brain booster",
    "celebrity secret", "click here", "guaranteed results", "shocking trick",
    "lock her up", "drag queen", "trans rights", "sexy singles",
    "seeking allah", "accept jesus", "eternal damnation", "child of god", 
    "according to the hadith"
]


soft_bad_words = [
    "free trial", "money back", "limited offer", "buy now", "get rich",
    "our top picks", "sponsored content", "terms and conditions",
    "as seen on", "white privilege", "systematic racism", "microaggression",
    "quran says", "the bible says","omnipresent", "omnipotent", "lgbtq", "casino",
    "patriarchy", "gender identity", "racial justice", "climate justice",
    "decolonize", "oppressor", "woke agenda", "cancel culture", "feminist theory",
]


hard_bad_pattern = re.compile(
    r'\b(?:' + '|'.join(re.escape(phrase) for phrase in hard_bad_words) + r')\b',
    flags=re.IGNORECASE
)


soft_bad_pattern = re.compile(
    r'\b(?:' + '|'.join(re.escape(phrase) for phrase in soft_bad_words) + r')\b',
    flags=re.IGNORECASE
)


##HELPER FUNCTIONS 
def remove_blacklist_content(text:str,threshhold=3):
    search_text     = text.lower()
    count           = 0 

    if hard_bad_pattern.search(text):
        return random.random() < .01
    else:
        for match in soft_bad_pattern.finditer(search_text):
            count += 1 

            if count > threshhold:
                return random.random() < .01

    return True


def count_whitelist_matches(text: str, phrases: list[str]) -> int:
    # Precompile regex with escaped phrases
    pattern = re.compile(r'(' + '|'.join(re.escape(p) for p in phrases) + r')', re.IGNORECASE)
    matches = pattern.findall(text)
    return len(matches)


def normalize_unicode_punctuation(text: str) -> str:
    # First, normalize the Unicode to NFKC form which often fixes quotes/dashes
    #text            = unicodedata.normalize("NFKC", text)

    # Manually replace common Unicode punctuation with ASCII
    single_replacements    = {
        '“': '"', '”': '"', '„': '"', '‟': '"',
        '‘': "'", '’': "'", '‚': "'", '‛': "'",
        '–': '-', '—': '-', '―': '-', '−': '-',
        '…': '...', '‒': '-','π':"pi",'…': '...',
        '•': '-', '·': '-', '・': '-', '∙': '-',
        '«': '"', '»': '"', '‹': '"', '›': '"',
        '½':"1/2",'⅓':"1/3",'⅔':"2/3", '¼':"1/4",
        '‹': '<', '›': '>', '«': '<<', '»': '>>',
        '‒': '-',
        '※': '*', '°': ' degrees',             # Optional expansion
        '\u00a0': ' ',  # non-breaking space
        '\u200b': '',   # zero-width space
        '\u200c': '',   # zero-width non-joiner
        '\u200d': '',   # zero-width joiner
        '\u202f': ' ',  # narrow no-break space
        '\u2060': '',   # word joiner
        '\u200e':'',
        '\u200f':'',
        '\u202a':'',
        '\u202b':'',
        '\u202c':'',
        '\u202d':'',
        '\u202e':''

    }
   
    #Replace all one-char items 
    newtext         = list(text)
    for i in range(len(newtext)):
        if newtext[i] in single_replacements:
            newtext[i] = single_replacements[newtext[i]] 

    # #Apply default fixes from language_utils.normalize_text
    text            = language_utils.normalize_text(text)

    return text


def passes_filter(article_text,hqp,keywords,article_i,thresh,sample_chance):
    check_text      = article_text.lower()

    if any([phrase in check_text for phrase in hqp]):
        return normalize_unicode_punctuation(article_text)
    else:
        hits            = 0 
        for phrase in keywords:
            hits        += check_text.count(phrase)
            if hits > thresh:
                return normalize_unicode_punctuation(article_text)
        else:
            if random.random() < sample_chance:
                return normalize_unicode_punctuation(article_text)
    return False 


def return_good_texts(data,thresh=3,sample_chance=.05,testing=False):
    fpath, keywords,hqp     = data 

    #Check if complete already
    cleaned_fpath:str       = fpath.replace(FINEWEB_CLEAN,TRAINING_TEXT)
    if os.path.exists(cleaned_fpath):
        return (0,0,0)
    
    text                    = open(fpath,'r',encoding='utf_8').read()
    articles                = text.split(END_TOKEN)

    good_articles           = [] 
    with multiprocessing.Pool(16) as pool:
        results             = pool.starmap(passes_filter,[(article,hqp,keywords,i,thresh,sample_chance) for i,article in enumerate(articles)])

        for good_article in results:
            if good_article:
                good_articles.append(good_article)
    
    with open(cleaned_fpath,'w',encoding='utf_8') as writefile:
        filetext            = f"{END_TOKEN}".join(good_articles) + END_TOKEN
        total_chars         = len(filetext)
        total_words         = len(filetext.split(" "))
        total_articles      = len(good_articles)
        writefile.write(f"{END_TOKEN}\n".join(good_articles) + END_TOKEN + "\n")

    return (total_chars,total_words,total_articles)


###FOR CODE 
import ast
import os
import re
from tqdm import tqdm

CORNERSTONE_IMPORTS = {
    'numpy', 'torch', 'pandas', 'scipy', 'matplotlib',
    'tensorflow', 'sklearn', 'socket', 'requests',
    'flask', 'http', 'argparse', 'sys', 'os', 're', 'logging'
}

SKIP_IMPORTS = {
    'ctypes', 'win32api', 'pywin32', 'comtypes', 'clr', 'windll', 'kernel32', 'user32', 'shcore'
}

SKIP_KEYWORDS = [
    'LoadLibrary', 'cdll', '.dll', 'win32', 'SetWindowLong', 'kernel32', 'FindWindow', 'GWL_EXSTYLE',
    'RegOpenKeyEx', 'RegisterClassEx', 'DllImport', 'CoInitialize', 'IUnknown', 'HRESULT',
]

SKIP_FILENAMES = {'setup.py', 'config.py', 'build.py', 'conftest.py', 'test_', 'tests'}


def is_clean_code(code: str) -> bool:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    name = n.name.split('.')[0]
                    if name in SKIP_IMPORTS:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    name = node.module.split('.')[0]
                    if name in SKIP_IMPORTS:
                        return False
        # Also scan for low-level Windows-style keywords
        for bad_kw in SKIP_KEYWORDS:
            if bad_kw in code:
                return False
        return True
    except Exception:
        return False


def uses_good_libs(code: str) -> bool:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if any(n.name.startswith(lib) for lib in CORNERSTONE_IMPORTS):
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(lib) for lib in CORNERSTONE_IMPORTS):
                    return True
        return False
    except Exception:
        return False


def is_valid_code_file(filename: str, code: str, max_lines=300, min_lines=10) -> bool:
    if any(skip in filename.lower() for skip in SKIP_FILENAMES):
        return False
    lines = code.strip().splitlines()
    return min_lines <= len(lines) <= max_lines


def filter_python_files(input_dir="clean_python_code", output_dir="filtered_code"):
    os.makedirs(output_dir, exist_ok=True)
    accepted = 0
    for fname in tqdm(os.listdir(input_dir)):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        if is_valid_code_file(fname, code) and uses_good_libs(code) and is_clean_code(code):
            with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as out:
                out.write(code)
            accepted += 1
    print(f"✅ Saved {accepted} cornerstone Python files.")


def contains_ml_keywords(code: str) -> bool:
    keywords = ['torch.optim','torch.nn.utils.','torch.nn','nn.Linear','nn.Conv','nn.Embedding']
    return any(kw in code for kw in keywords)


def filter_python(data:pandas.DataFrame):
    year    = data['max_issues_repo_issues_event_min_datetime']

"""
Cleans the existing data found in training.FINEWEB_BASE.
 - removes content based on language score, total length, language, and blacklist criteria 
 - saves content to training.FINEWEB_CLEAN  
"""
def clean_fineweb(min_score=.97,use_stream=False):

    texts       = [] 

    paths       = [os.path.join(FINEWEB_BASE,fpath) for fpath in os.listdir(FINEWEB_BASE) if ".parquet" in fpath]
    random.shuffle(paths)
    fpaths      = tqdm.tqdm(paths)

    for file in fpaths:

        curfile     = file.replace(FINEWEB_BASE,FINEWEB_CLEAN).replace(".parquet",".txt")
        if os.path.exists(curfile):
            continue

        data        = pandas.read_parquet(file,engine='pyarrow')
        for t,s,l in zip(data['text'],data['language_score'],data['language']):

            if l == 'en' and s > min_score and len(t) > 2_500 and remove_blacklist_content(t):
                texts.append(t + "\n" + END_TOKEN+"\n")

        
        with open(curfile,'w',encoding='utf_8') as curwrite:
            curwrite.write("".join(texts))
            curwrite.close()

        #Reset texts
        texts       = []


def match_keys(title:str):
    return title.lower().replace(" ","").replace("_","")

#Downloaded from
#https://dumps.wikimedia.org/other/pageviews/2024/2024-02/
def generate_freq(cutoff=200):

    fnames  = [f"F:/data/pageviews{i}.txt" for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]

    data    = {}

    MEDIAN_MULT = 2

    for fname in fnames:
        contents    = open(fname,'r',encoding='utf_8').readlines()

        #Filter only english, non-files 
        lines       = [l for l in contents if l[:3] == 'en ' and not ' file:' in l.lower() and not ' category:' in l.lower() and not ' list_of_' in l.lower()] 
        #Get object name 
        items       = [line.split(" ") for line in lines]
        
        for item in items:
            obj     = match_keys(item[1])
            count1  = int(item[2])

            if obj in data:
                data[obj].append(count1) 
            else:
                data[obj] = [count1]

    filtered_stats  = {}
    for key in data:

        stats   = data[key]

        #if its less than 3 entries, then no shot
        if len(stats) < 5:
            continue

        median  = int(numpy.median(stats))

        smoothed= numpy.asarray([min(s,2*median) for s in stats])

        score   = 3*(1 / (.1 + numpy.std(smoothed))) + 2*(sum(smoothed) / len(smoothed)) * 2*(len(smoothed) ** .333) 
        if score > cutoff:
            filtered_stats[key] = score


    # for item in filtered_stats:
    #     if filtered_stats[item] < cutoff:
    #         continue
    #     input(f"{item} -> {filtered_stats[item]}")
    # #Sort it     
    return filtered_stats


def stream_wiki(min_counts=30):
    """
    Generator that yields (title, text) tuples from Wikipedia XML dump,
    skipping redirects and empty articles.
    """

    stats       = generate_freq()
    print(f"generated stats")
    # Use iterparse with events for efficient streaming
    with open("E:/data/wiki.xml", 'r',encoding='utf_8') as file:
        context = ET.iterparse(file, events=('start', 'end'))
        _, root = next(context)  # get root element

        page_data = {}
        inside_text = False

        for event, elem in context:
            tag = elem.tag.split('}')[-1]  # strip namespace

            if event == 'start':
                if tag == 'page':
                    page_data = {'title': '', 'text': '', 'redirect': False}
                elif tag == 'text':
                    inside_text = True

            elif event == 'end':
                if tag == 'title':
                    page_data['title'] = elem.text or ''
                elif tag == 'redirect':
                    page_data['redirect'] = True
                elif tag == 'text':
                    page_data['text'] = elem.text or ''
                    inside_text = False
                elif tag == 'page':
                    if not page_data['redirect'] and page_data['text'].strip():
                        title   = match_keys(page_data['title'])
                        text    = clean_article(page_data['text'])
                        #If not title in stats, skip
                        if not title in stats:
                            continue
                        elif not stats[title] > min_counts or len(text) < 1000:
                            continue 
                        else:
                            yield (title, stats[title], len(text), text)

                    root.clear()  # Free memory


def clean_fineweb_streamed(min_score=.97):
    texts       = [] 

    cur_i       = 0 
    fname       = f"{FINEWEB_CLEAN}/2024_38_{cur_i}.txt"
    while os.path.exists(fname):
        cur_i += 1 
        fname       = f"{FINEWEB_CLEAN}/2024_38_{cur_i}.txt"
    
    queue       = [] 

    for newtext in stream_dataset():

        if newtext['language'] == 'en' and newtext['language_score'] > min_score and len(newtext['text']) > 2_500:
            queue.append(newtext['text'] + END_TOKEN)

        if len(queue) > 100_000:

            with multiprocessing.Pool(12) as pool:
                results     = pool.map(remove_blacklist_content,queue)

                for result,text in zip(results,queue):
                    if result:
                        texts.append(text)
            queue   = [] 

        if len(texts) > 150_000:
            with open(fname,'w',encoding='utf_8') as writefile:
                writefile.write("".join(texts))
            
            texts   = [] 
            cur_i += 1
            fname       = f"{FINEWEB_CLEAN}/2024_38_{cur_i}.txt"


import re
def clean_article(article_text: str) -> str:
    """
    Cleans raw Wikipedia wikitext into plain, readable text.
    Removes formatting, templates, references, and markup.
    """

    # Remove comments: <!-- comment -->
    article_text = re.sub(r'<!--.*?-->', '', article_text, flags=re.DOTALL)

    # Remove file/image links: [[File:...]] or [[Image:...]]
    article_text = re.sub(r'\[\[(File|Image):.*?\]\]', '', article_text, flags=re.IGNORECASE)

    # Remove templates: {{...}}, including nested ones (approximate)
    article_text = re.sub(r'\{\{[^{}]*\}\}', '', article_text)

    # Remove reference tags: <ref>...</ref> and self-closing <ref ... />
    article_text = re.sub(r'<ref[^>/]*?>.*?</ref>', '', article_text, flags=re.DOTALL)
    article_text = re.sub(r'<ref[^>]*/>', '', article_text)

    # Remove HTML tags like <div>, <span>, <br />, etc.
    article_text = re.sub(r'</?\w+[^>]*?>', '', article_text)

    # Unwrap internal links: [[Link|display]] -> display, [[Link]] -> Link
    article_text = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', article_text)

    # Remove external links: [http://example.com label] -> label, [http://example.com] -> ''
    article_text = re.sub(r'\[https?:\/\/[^\s\]]+(\s+([^\]]+))?\]', lambda m: m.group(2) if m.group(2) else '', article_text)

    # Decode some HTML entities
    article_text = article_text.replace("&nbsp;", " ")
    article_text = article_text.replace("&amp;", "&")
    article_text = article_text.replace("&lt;", "<")
    article_text = article_text.replace("&gt;", ">")
    article_text = article_text.replace("''",'"')

    # Remove category links: [[Category:XYZ]]
    article_text = re.sub(r'\[\[Category:[^\]]+\]\]', '', article_text)

    # Remove language links and interwiki: [[fr:XYZ]], [[de:XYZ]]
    article_text = re.sub(r'\[\[[a-z\-]+:[^\]]+\]\]', '', article_text, flags=re.IGNORECASE)

    # Collapse multiple newlines into one or two (for paragraphing)
    article_text = re.sub(r'\n{3,}', '\n\n', article_text)
    article_text = re.sub(r'[ \t]+\n', '\n', article_text)

    while "*\n*" in article_text:
        article_text = article_text.replace("*\n*","*")

    # Final strip
    return article_text.strip()


"""
Cleans the existing data found in training.FINEWEB_CLEAN.
 - passes all content that has enough hits in keywords or high_quality_phrases  
 - saves content to training.FINEWEB_CLEAN  
"""
def filter_by_topic(text_root:str=f"{FINEWEB_CLEAN}"):#,tokenizer:ByteLevelBPETokenizer=None):

    keywords = list(set([
        # --- AI / Machine Learning / NLP ---
        "machine learning", "deep learning", "artificial intelligence", "neural network",
        "transformer model", "language model", "natural language processing", "llm",
        "reinforcement learning", "supervised learning", "unsupervised learning",
        "fine-tuning", "prompt engineering", "transfer learning", "few-shot learning",
        "backpropagation", "gradient descent", "vector embeddings", "attention mechanism",
        "tokenization", "pos tagging", "named entity recognition", "text classification",
        "openai", "chatgpt", "mistral ai", "gemini model", "claude model", "anthropic",
        "hugging face", "pytorch", "tensorflow", "jax", "transformers library", "onnx",
        "diffusion model", "stable diffusion", "generative ai", "multimodal model",

        # --- Programming / CS / Dev Tools ---
        "python", "c++", "c#", "rust", "golang", "typescript", "javascript",
        "html5", "css3", "bash scripting", "unix shell", "command line",
        "react", "vuejs", "nodejs", "expressjs", "fastapi", "flask",
        "data structures", "algorithms", "recursion", "binary tree", "hash map",
        "compiler design", "systems programming", "assembly language", "llvm",
        "concurrency", "parallel processing", "distributed systems", "sockets",
        "api development", "web scraping", "microservices", "graphql",

        # --- DevOps / Cloud / Infra ---
        "docker", "kubernetes", "ci/cd", "infrastructure as code", "ansible",
        "devops", "terraform", "aws lambda", "azure cloud", "gcp compute",
        "serverless architecture", "cloud computing", "linux administration",

        # --- Data Science / Analysis ---
        "data science", "data visualization", "data mining", "big data",
        "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "plotly",
        "sql", "database indexing", "postgresql", "mongodb", "data warehouse",
        "feature engineering", "dimensionality reduction", "anomaly detection",

        # --- Math / Theoretical CS ---
        "abstract algebra", "differential equations", "group theory",
        "manifold", "calculus of variations", "fourier transform",
        "laplace transform", "symmetric matrix", "markov chain",
        "expected value", "entropy in information theory", "computability",
        "lambda calculus", "godel incompleteness", "kolmogorov complexity",
        "information entropy", "graph isomorphism", "diophantine equations",
        "linear algebra", "calculus", "derivative", "integral", "function"

        # --- Physics / Engineering ---
        "general relativity", "special relativity", "higgs boson", "standard model of particle physics",
        "quantum entanglement", "quantum tunneling", "schrodinger equation",
        "maxwell's equations", "laplace's equation", "navier-stokes equations",
        "electromagnetic induction", "nuclear fusion", "thermodynamic cycles",
        "phase transition", "materials science", "mechanical engineering principles",
        "acoustics", "vibration analysis", "aerospace engineering", "engineering statics",

        # --- Chemistry / Biology ---
        "chemical bonding", "reaction kinetics", "molecular orbitals",
        "mass spectrometry", "protein folding", "enzyme kinetics",
        "rna transcription", "dna replication", "mitosis and meiosis",
        "neurotransmitters", "brain plasticity", "immune response mechanisms",
        "biomechanics", "photosynthetic pathways", "molecular biology techniques",

        # --- Astronomy / Earth Science ---
        "n-body problem", "heliocentric model", "redshift and blueshift",
        "kepler's laws", "planck length", "gravity waves",
        "solar flare", "magnetosphere", "earth magnetic field reversal",
        "geochronology", "igneous rock formation", "volcanic activity",
        "tectonic plate boundary", "seismology", "paleoclimatology",

        # --- Music Theory / Musicology ---
        "counterpoint", "harmonic progression", "circle of fifths",
        "equal temperament", "baroque composition", "modal scales",
        "atonality", "serialism", "tonal center", "polyrhythm",
        "syncopation", "fugue structure", "dynamic expression",
        "ear training", "score analysis", "chord substitution",
        "orchestration techniques", "musical phrasing",

        # --- History / Philosophy / Civics ---
        "political enlightenment", "aristotelian logic", "dialectical materialism",
        "existentialist philosophy", "stoic ethics", "lockean liberalism",
        "kantian imperative", "utilitarian calculus", "social contract theory",
        "republicanism vs monarchy", "constitutional convention", "alexander hamilton",
        "treaty of versailles", "world systems theory", "decolonization in africa",
        "great man theory", "historical materialism", "economics", "history",

        # --- Economics / Business ---
        "keynesian economics", "austrian school", "quantity theory of money",
        "game theory", "coase theorem", "gini coefficient", "marginal utility",
        "opportunity cost", "price elasticity", "economic equilibrium",
        "efficient market hypothesis", "risk-adjusted return", "net present value",
        "strategic market segmentation", "unit economics", "founder-market fit",
        "value proposition design", "econometrics model", "economics",

        # --- Cross-Disciplinary Concepts ---
        "emergent complexity", "information theory", "cybernetics",
        "systems thinking", "agent-based modeling", "scientific revolution",
        "scientific reproducibility", "statistical significance",
        "longitudinal study", "knowledge representation", "rational decision making",
        "game theory",

        # --- Content Creation / YouTube / Streaming ---
        "youtube algorithm", "youtube analytics", "youtube monetization",
        "audience retention", "click-through rate", "content creator",
        "obs studio", "streaming setup", "video editing", "adobe premiere",
        "davinci resolve", "final cut pro", "podcasting", "thumbnail design",
        "script writing", "content calendar", "youtube shorts", "tiktok strategy",
        "sponsor deal", "affiliate marketing", "community engagement", "growth hacking",
        "iced coffee hour", "ben shapiro show", "coffeezilla", "harris heller",

        # --- Ethics / Society / Tech Policy ---
        "ai safety", "alignment problem", "bias in ai", "social media ethics",
        "surveillance capitalism", "data privacy", "gdpr", "net neutrality",
        "open source licensing", "mit license", "agpl", "freedom of information",

        # Meaning, Purpose, and Identity
        "meaning of life", "search for meaning", "personal responsibility",
        "identity crisis", "self-authoring", "call to adventure", "life purpose",
        "individuation", "hero's journey", "finding purpose", "sense of direction",
        "psychological integration", "moral compass",

        # Philosophical and Theological
        "existentialism", "stoicism", "virtue ethics", "logos", "telos",
        "mythological structure", "symbolic meaning", "religious symbolism",
        "archetype", "narrative truth", "metaphysical order", "chaos and order",
        "sacred patterns", "transcendence", "divine hierarchy",

        # Psychology and Self-Improvement
        "shadow integration", "jungian psychology", "freudian analysis",
        "cognitive restructuring", "discipline equals freedom", "hierarchy of competence",
        "resilience", "grit", "voluntary suffering", "delayed gratification",
        "conscientiousness", "truth over comfort", "responsibility before rights",
        "face your fears", "clean your room", "make your bed", "order your life",
        "narrative identity", "post-traumatic growth",

        # Conversations and Dialogues
        "long-form discussion", "open-ended conversation", "meaningful conversation",
        "deep dialogue", "truth-seeking discussion", "civil discourse",

        # Philosophical Questions
        "what is truth", "what is good", "what is evil", "free will vs determinism",
        "moral responsibility", "objective morality", "the nature of consciousness",

        # Influencer/Podcaster References
        "jordan peterson", "joe rogan experience", "jonathan pageau", "cosmic skeptic",
        "lex fridman podcast", "huberman lab", "modern wisdom", "eric weinstein",
        "meaning crisis", "veritasium", "thomas sowell", "sam harris", "ben shapiro",

        # Ancient Wisdom and Stoicism
        "meditations by marcus aurelius", "letters from a stoic", "plato's republic",
        "the cave allegory", "aristotle virtue", "socratic method", "eastern philosophy",
        "tao te ching", "buddhist mindfulness", "zen koan", "non-dualism",

        # Contemplation of Death / Becoming
        "memento mori", "face mortality", "life is suffering", "transformation through struggle",
        "death awareness", "becoming who you are", "birth of the self", "transcend the ego"
    ]))
    
    high_quality_phrases = list(set([
        # Philosophy, Reflection, Thought
        "what does it mean to",
        "the nature of",
        "one of the most important ideas",
        "at the core of this idea is",
        "this raises the question of",
        "a deeper understanding of",
        "if we take a step back",
        "the underlying principles of",
        "in search of meaning",
        "the essence of",
        "a moral responsibility to",
        "an existential question",
        "exploring the human condition",
        "truth is often found in",
        "the philosophical implications",
        "the way we see the world",

        # Science, Analysis, and Reasoning
        "from a scientific point of view",
        "in order to understand",
        "this model suggests that",
        "the data indicates",
        "through trial and error",
        "a predictive framework for",
        "from first principles",
        "empirical evidence shows",
        "according to recent studies",
        "a testable hypothesis",
        "scientific consensus on",
        "the mechanism behind",
        "a system-level perspective",
        "validated through observation",

        # Personal Development & Mindset
        "growth often comes from",
        "to become the best version of",
        "the discipline required to",
        "navigating uncertainty",
        "mental models we rely on",
        "what we can learn from failure",
        "the habits that shape",
        "meaningful change begins with",
        "taking ownership of",
        "deliberate practice leads to",
        "long-term thinking requires",

        # Tech & Innovation
        "the future of technology",
        "a breakthrough in",
        "exponential growth of",
        "the architecture of modern systems",
        "challenges in scaling",
        "this changes how we think about",
        "paradigm shift in",
        "human-machine collaboration",
        "technological disruption in",
        "open source movement has enabled",

        # History / Civilization / Culture
        "over the course of history",
        "the legacy of",
        "from ancient times to",
        "the fall and rise of",
        "a turning point in history",
        "civilizations have risen and fallen",
        "patterns across time",
        "historical context reveals",
        "ww2", "ww2", "word war"

        # Blogging / Editorial Framing
        "in this post we'll explore",
        "as someone who has worked in",
        "after years of experience",
        "what I've learned from",
        "here's why that matters",
        "this article examines",
        "let's take a closer look",
        "in my experience",
        "breaking this down step by step",
        "if you're wondering why",

        # Longform Journalism
        "according to multiple sources",
        "in an exclusive interview",
        "an in-depth analysis of",
        "revealed documents show",
        "reporting from the ground",
        "sources close to the matter",
        "confirmed by independent outlets",
        "in recent developments",
        "new evidence suggests",
        "investigative journalists uncovered",

        # Analytical and Reflective Tone
        "when you think about it",
        "this leads to a broader question",
        "the bigger picture here is",
        "connecting the dots between",
        "we often overlook the fact that",
        "at the intersection of",
        "the key takeaway is",
        "framed through the lens of",
        "based on these observations",
        "this pattern can be seen across",

        # Newsletter / Substack / Thoughtful Commentary
        "subscribe to get future posts",
        "as I wrote in my last piece",
        "let me walk you through",
        "if this resonates with you",
        "insights from my newsletter",
        "thank you for reading this far",
        "you can support my work by",
        "a few thoughts on",
        "recapping this week's events",
        "published via substack"
    ]))

    fpaths              = [(os.path.join(text_root,fpath),keywords,high_quality_phrases) for fpath in os.listdir(text_root)]
    fpaths              = tqdm(fpaths)

    results             = [] 
    for path in fpaths:
        results.append(return_good_texts(path))

    # with multiprocessing.Pool(16) as pool:
    #     results         = pool.map(return_good_texts,args)

    chars               = 0 
    words               = 0
    articles            = 0
    for result in results:
        c,w,a           = result 
        chars += c 
        words += w 
        articles += a
        

    print(f"Generated dataset of:\n\tchars:\t{chars//1_000_000}M\n\twords:\t{words//1_000_000}M58\n\tpages:\t{articles}")


def filter_wikipedia():
    texts       = [] 

    cur_i       = 0 
    fname       = f"{TRAINING_TEXT}/wikipedia_{cur_i}.txt"
    while os.path.exists(fname):
        cur_i += 1 
        fname       = f"{TRAINING_TEXT}/wikipedia_{cur_i}.txt"

    for article in stream_wiki():
        texts.append(article[0] + "\n\n" + article[-1] + END_TOKEN)
        if len(texts) > 20_000:
            with open(fname,'w',encoding='utf_8') as writefile:
                writefile.write(f"".join(texts))
            
            cur_i += 1
            fname       = f"{TRAINING_TEXT}/wikipedia_{cur_i}.txt"
            texts = []
    
    with open(fname,'w',encoding='utf_8') as writefile:
        writefile.write(f"".join(texts))
        
        cur_i += 1
        fname       = f"{TRAINING_TEXT}/wikipedia_{cur_i}.txt"
        texts = []


def runit():
    for fname in os.listdir("D:/nlp/traintext2"):
        fname = os.path.join("D:/nlp/traintext2",fname)

        os.rename(fname,fname.replace('python','python_full'))


if __name__ == '__main__':
    f = generate_freq()
    with open("//Steinpc/s/nlp/data/article_scores.json",'w',encoding='utf-8') as writefile:
        writefile.write(json.dumps(f))
    exit()
    runit()
    exit()
    dd = [] 
    counter = 0 
    queue = [] 
    nruns   = 100_000
    streamer = stream_dataset()
    streamer.__next__()

    t0 = time.time()
    for newtext in streamer:

        if remove_blacklist_content(newtext['text']):
            dd.append(newtext['text'])

        counter += 1 

        if counter > nruns:
            break 
    
    
    print(f"{nruns} by individual in {time.time() - t0}")

    t0 = time.time()
    counter = 0 
    dd = [] 
    for newtext in streamer:

        queue.append(newtext['text'])

        if len(queue) == 20_000:
            with multiprocessing.Pool(8) as pool:

                results     = pool.imap(remove_blacklist_content,queue)

                for result,text in zip(results,queue):
                    if result:
                        dd.append(text)

            queue = [] 


        counter += 1 

        if counter > nruns:
            break 
    
    
    print(f"{nruns} by queue in {time.time() - t0}")
