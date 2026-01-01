import tqdm 
import os 
import json
from tokenizers.implementations import ByteLevelBPETokenizer
import crawl 
from training import *
import numpy
import time 
import multiprocessing
import random 
from dataset import TokenizedDataset, load_tokenizer
from collections import Counter
import unicodedata
import re 


#Tokenizes based on the tokens found in CRAWL_DB
def train_tokenizer(vocab_size:int,name:str,db:str=CRAWL_DB) ->ByteLevelBPETokenizer:
    print(f"Training {name} tokenizer size={vocab_size}")
    tokenizer               = ByteLevelBPETokenizer()
    tokenizer.train([os.path.join(db,fname) for fname in os.listdir(db)],vocab_size=vocab_size-1)
    tokenizer.add_tokens([END_TOKEN])

    if not os.path.exists(f"{PATH}/{name}"):
        os.mkdir(f"{PATH}/{name}")

    tokenizer.save_model(f"{PATH}/{name}")
    print(f"\tcomplete - saved as {name}")
    

#Loads tokenizer from default location. Adds the endoftext token
def load_tokenizer(tokenizer_name:str)->ByteLevelBPETokenizer:
    tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename=f"{tokenizer_name}/vocab.json",merges_filename=f"{tokenizer_name}/merges.txt")
    tokenizer.add_tokens([END_TOKEN])
    return tokenizer



def tokenize_save_text(data:dict):

    #Load contents
    tokenizer   = data['tokenizer']
    fpath       = data['fpath']
    db          = data['db']
    tok_db      = data['tok_db']

    tokenizer   = load_tokenizer(tokenizer)
    with open(fpath,'r',encoding='utf_8') as readfile:
        contents    = readfile.read()

        #Split to all pages 
        webpages        = contents.split(END_TOKEN)
        encoded_pages   = tokenizer.encode_batch(webpages)
        encoded_pages   = [numpy.asarray(page.ids + tokenizer.encode(END_TOKEN).ids,dtype=numpy.uint16) for page in encoded_pages]
        np_ids          = numpy.concatenate(encoded_pages)

        # print(f"tokenizing {fpath}")
        # ids         = tokenizer.encode(contents).ids
        np_ids      = np_ids.astype(numpy.uint16)
        tokpath     = fpath.replace(db,tok_db).replace(".txt",".npy")
        numpy.save(tokpath,np_ids)
        tot_tokens      = len(np_ids)
        del np_ids


    return tot_tokens


#Tokenizes the corpus found in CRAWL_DB and saves it to TOK_DB
def tokenize_corpus(tokenizer_name:str,db:str=TRAINING_TEXT,tok_db:str=TRAINING_TOKENS,n_workers:int=4):
    print(f"tokenizing corpus")
    corpus      = [os.path.join(db,fname) for fname in os.listdir(db)]

    args        = [] 

    for fpath in corpus:
        tokpath     = fpath.replace(db,tok_db).replace(".txt",".npy")

        #Skip if we've done it
        if os.path.exists(tokpath):
            continue

        args.append(({'tokenizer':f"{PATH}/{tokenizer_name}",'fpath':fpath,'db':db,'tok_db':tok_db}))
    
    with multiprocessing.Pool(processes=n_workers) as pool:
       results      = pool.map(tokenize_save_text,args)
    

    total_tok   = 0 
    for res in results:
        total_tok += res 

    print(f"generated {total_tok/1_000_000_000:.3f}B tokens")


def load_tokens(args,max_tokens):
       #Load data 
    tokens                      = [] 
    n_tok_loaded                = 0
    fnames                      = [fname for fname in os.listdir(f"{args.train_root}/{args.ds_name}")]
    fnames.sort(key= lambda x: int(x.replace("tokens","").replace(".npy","").replace(".txt","")))
    for fname in fnames:
        fname               = f"{args.train_root}/{args.ds_name}/{fname}"
        newtok:numpy.array  = numpy.load(fname).astype(numpy.uint16)
        tokens.append(newtok)
        n_tok_loaded        += len(newtok)

        if n_tok_loaded > max_tokens:
            break

    tokens                      = numpy.concatenate(tokens)[-n_tok_loaded:]
    dataset                     = TokenizedDataset(tokens,eval(args.input_size))
    _N_TOKENS                   = dataset.n_tokens

    return dataset,len(tokens)


def normalize_unicode_punctuation(text: str) -> str:
    # First, normalize the Unicode to NFKC form which often fixes quotes/dashes
    text = unicodedata.normalize("NFKC", text)

    #fix contents
    for rep_word in ["the", "and", " is", "are", "of"]:
        text    = text.replace(f" {rep_word} {rep_word} ", f" {rep_word} ")


    # Manually replace common Unicode punctuation with ASCII
    replacements = {
        '“': '"', '”': '"', '„': '"', '‟': '"',
        '‘': "'", '’': "'", '‚': "'", '‛': "'",
        '–': '-', '—': '-', '―': '-', '−': '-',
        '…': '...', '‒': '-',
        '•': '-', '·': '-', '・': '-', '∙': '-',
        '«': '"', '»': '"', '‹': '"', '›': '"',
        '※': '*', '°': ' degrees',  # Optional expansion
        '\u00a0': ' ',  # Non-breaking space to regular space
        '\u200b': '',   # Zero-width space removed
        '\u202f': ' ',  # Narrow non-breaking space
    }

    # Replace based on the mapping
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # Remove any lingering weird control characters
    text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)

    return text


def save_good_texts(data,thresh=3,sample_chance=.05):

    fpath, keywords,hqp     = data 

    #Check if complete already
    cleaned_fpath:str       = fpath.replace(FINEWEB_CLEAN,TRAINING_TEXT)
    if os.path.exists(cleaned_fpath):
        return (0,0,0)
    
    #Load articles and prep for cleaning
    articles                = open(fpath,'r',encoding='utf_8').read().split(END_TOKEN)
    articles                = [normalize_unicode_punctuation(article) for article in articles]

    check_articles          = [article.lower() for article in articles]
    good_articles           = [] 

    for article,check_article in zip(articles,check_articles):

        if any([phrase in check_article for phrase in hqp]):
            good_articles.append(article)
        else:
            hits            = 0 
            for phrase in keywords:
                hits        += check_article.count(phrase)
                if hits > thresh:
                    good_articles.append(article)
                    break
            else:
                if random.random() < sample_chance:
                    good_articles.append(article)
        

    #Save and write file
    with open(cleaned_fpath,'w',encoding='utf_8') as writefile:
        filetext            = f"{END_TOKEN}".join(good_articles) + END_TOKEN
        total_chars         = len(filetext)
        total_words         = len(filetext.split(" "))
        total_articles      = len(good_articles)
        writefile.write(f"{END_TOKEN}".join(good_articles) + END_TOKEN)

    return (total_chars,total_words,total_articles)



#Finds all texts within corpus that contain keywords
def filter_by_topic(text_root:str=f"{FINEWEB_CLEAN}"):#,tokenizer:ByteLevelBPETokenizer=None):

    print(f"filtering by keywords")
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
    
    high_quality_phrases = [
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
    ]

    fpaths              = [(os.path.join(text_root,fpath),keywords,high_quality_phrases) for fpath in os.listdir(text_root)]
    args                = fpaths

    results             = [] 
    for path in fpaths:
        results.append(save_good_texts(path))

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
        

    print(f"Generated dataset of:\n\tchars:\t{chars}\n\twords:\t{words}\n\tpages:\t{articles}")




if __name__ == "__main__":
    name        = f"{PATH}/32k_c++"

    #train_tokenizer(32768,name,db=FINEWEB_CLEAN)
    #t = load_tokenizer(name)
    #print(f"Steinshark -> {t.encode('Steinshark').ids} -> {t.decode(t.encode('Steinshark').ids)}")
    #exit()
    #Loop so that it always runs
    #while True:
    filter_by_topic(text_root=FINEWEB_CLEAN)#,tokenizer=t)
    #tokenize_corpus(name,db=FINEWEB_CLEAN,tok_db=TOK_DB_CLEAN,n_workers=8)
        #time.sleep(10)