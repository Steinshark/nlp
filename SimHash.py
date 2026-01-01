# Author: Everett Stenberg (Steinshark)
import zlib 
import numpy
import xxhash
from collections import Counter 
import os 
from training import END_TOKEN
import random
import json 

#Subject to change 
hash_fn         = xxhash.xxh3_64
hash_len        = 64
top_k_remove    = 100


def clean_doc(contents:str):
    #Lower 
    contents    = contents.lower()
    
    for item in [".", "?","!", "\n", "-"]:
        contents    = contents.replace(item, " ")

    for item in ['"',"'",')','(','[',']']:
        contents    = contents.replace(item,"")

    #Remove whitespace idiocy
    while "  " in contents:
        contents = contents.replace("  ", " ")
    while "\n\n" in contents:
        contents = contents.replace("\n\n", "\n")

    return contents
    
    
def word_doc_counter(ds_path:str,stop_after:int):
    
    term_freqs      = {}
    n_docs          = 0
    breakout        = False

    db_files        = os.listdir(ds_path)
    random.shuffle(db_files)

    for document in db_files:
        if not ".wetc" in document:
            pass
        try:
            contents         = open(os.path.join(ds_path,document),'r',encoding='utf-8').read()
        except FileNotFoundError:
            continue 

        documents        = contents.split(END_TOKEN)

        for doc in documents:
            contents         = clean_doc(doc)
            words            = contents.split(" ")
            counts           = Counter(words)
            n_docs           += 1

            for word in counts:
                if word in term_freqs:
                    term_freqs[word] += counts[word]
                else:
                    term_freqs[word] = counts[word]

            if n_docs > stop_after:
                breakout = True 
                break

        if breakout:
            break 
    wordcounts  = [(c,w) for w,c in term_freqs.items()]
    wordcounts  = sorted(wordcounts,key= lambda x: x[0],reverse=True)
    stopwords   = set([item[1] for item in wordcounts[:top_k_remove]])

    return stopwords


def tf_idf(term,term_count,n_terms,counts,n_docs):
    tf                 = term_count / n_terms 

    if term in counts:
        count   = counts[term]
    else:
        count   = 1 

    idf                = numpy.log(n_docs / 1)

    return tf * idf
    
    
#This function processess a single document for SimHash analysis 
def simhash(doc:str,stopwords:list[str]):
    
    #Tokenize
    doc_tokens      = tokenize_document(doc,stopwords)
    if doc_tokens is None:
        return None
    #Hash 
    document_hash   = hash_document(doc_tokens)
    return document_hash


#Convert the hash (intdigest form) into a bitarray
def bit_arr(hash_digest:int):
    bit_arr     = [0 for _ in range(hash_len)]
    
    for i in range(hash_len):
        bit_arr[i] = (hash_digest >> i) & 1 
    
    return numpy.asarray(bit_arr) * 2 - 1 
    
    
#This function cononacalizes the contents of a document
#and returns an array of scaled tokens based on word count (log count) 
def tokenize_document(doc:str,stopwords:list[str]):
    
    #Lower 
    working_text    = clean_doc(doc)
    
    #Split and count words 
    working_words   = [w for w in working_text.split(" ") if not w in stopwords]
    if len(working_words) < 1:
        return None
    
    word_counts     = Counter(working_words)
    log_word_counts = {word: numpy.log(count) for word,count in word_counts.items()}
    
    #Get tokens
    encoded_words   = [word.encode() for word in working_words]
    tokens          = [bit_arr(word.intdigest()) for word in map(hash_fn,encoded_words)]
    
    #Scale each token by log TF
    weighted_tokens = []
    for token, word in zip(tokens,working_words):
        weighted_token = token * log_word_counts[word]
        weighted_tokens.append(weighted_token)
    
    return weighted_tokens
    tokens_array    = numpy.stack(tokens)
    weights         = numpy.array([log_word_counts[word] for word in working_words])
    weighted_tokens = tokens_array * weights[:, None]

    
    return weighted_tokens


#This function hashes the document based on the weighted tokens 
def hash_document(weighted_tokens:numpy.array):
    
    #Accumulate weights of document based on each token's contribution
    accu_vector = numpy.zeros((hash_len,))
    for token in weighted_tokens:
        accu_vector += token

    #Determine document hash
    document_hash = numpy.zeros((hash_len,))
    for i in range(len(accu_vector)):
        document_hash[i] = int(accu_vector[i] > 0)
    
    return document_hash

def get_stopwords():
    stopwords_file  = open("simhash_stopwords.txt",'r',encoding='utf_8').read()

    stopwords       = json.loads(stopwords_file)

    return stopwords

if __name__ == "__main__":
    import time
    import json 

    stopwords   = word_doc_counter("D:\\nlp\\crawl\\",100_000)
    f           = open("simhash_stopwords.txt",'w',encoding='utf_8')
    f.write(json.dumps(list(stopwords) ))
    f.close()
    exit()
    files   = [open(os.path.join("D:\\nlp\\crawl\\",p),'r',encoding='utf-8').read() for p in [d for d in os.listdir("D:\\nlp\\crawl\\")if '.wetc' in d]]
    documents = []
    for doc in files:
        documents += doc.split(END_TOKEN)

    t0 = time.time()
    for i,doc in enumerate(documents):

        tokenized = simhash(doc,stopwords)

        if i % 1000 == 0:
            print(str(i) + " " + str(time.time()-t0))

            t0 = time.time()
 
