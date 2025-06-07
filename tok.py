import os 
import json
from tokenizers.implementations import ByteLevelBPETokenizer
import crawl 
from training import *
import numpy
import time 

#Tokenizes based on the tokens found in CRAWL_DB
def train_tokenizer(vocab_size:int,name:str) ->ByteLevelBPETokenizer:
    print(f"Training {name} tokenizer size={vocab_size}")
    tokenizer               = ByteLevelBPETokenizer()
    tokenizer.train([os.path.join(CRAWL_DB,fname) for fname in os.listdir(CRAWL_DB)],vocab_size=vocab_size-1)
    tokenizer.add_tokens([END_TOKEN])

    if not os.path.exists(f"{PATH}/{name}"):
        os.mkdir(f"{PATH}/{name}")

    tokenizer.save_model(f"{PATH}/{name}")
    print(f"\tcomplete - saved as {name}")
    

#Loads tokenizer from default location. Adds the endoftext token
def load_tokenizer(tokenizer_name:str)->ByteLevelBPETokenizer:
    tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename=f"{PATH}/{tokenizer_name}/vocab.json",merges_filename=f"{PATH}/{tokenizer_name}/merges.txt")
    tokenizer.add_tokens([END_TOKEN])
    print(f"loaded tokenizer size {tokenizer.get_vocab_size()}")
    return tokenizer


#Tokenizes the corpus found in CRAWL_DB and saves it to TOK_DB
def tokenize_corpus(tokenizer_name:str):
    tokenizer   = load_tokenizer(tokenizer_name)

    corpus      = [os.path.join(CRAWL_DB,fname) for fname in os.listdir(CRAWL_DB)]

    for fpath in corpus:
        tokpath     = fpath.replace(CRAWL_DB,TOK_DB).replace(".txt",".npy")

        #Skip if weve done it
        if os.path.exists(tokpath):
            continue

        #Skip if its less than filesize 
        if os.path.getsize(fpath) < 130000000:
            continue 
        else:
            #If its over, give it 30 seconds in case its the one being finalized
            time.sleep(10)


        print(f"\ttokenizing {fpath.replace(CRAWL_DB,"")}")
        
        with open(fpath,'r',encoding='utf_8') as readfile:
            text    = readfile.read()
        
        ids         = tokenizer.encode(text).ids
        tokens      = numpy.asarray(ids).astype(numpy.int16)
        

        numpy.save(tokpath,tokens)

        


if __name__ == "__main__":
    name        = "32k"
    #train_tokenizer(32768,name)
    t = load_tokenizer(name)
    print(f"Steinshark -> {t.encode('Steinshark').ids} -> {t.decode(t.encode('Steinshark').ids)}")
    exit()
    #Loop so that it always runs
    while True:
        tokenize_corpus(name)
        time.sleep(10)