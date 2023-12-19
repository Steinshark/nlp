import torch 
from torch.utils.data import Dataset
import os 
from transformers import PreTrainedTokenizer
import random
import re
import nltk
from nltk import corpus
from model import GPTSteinsharkTokenizer 
import time 

def load_data(ds_root:str,chunk_size:int,tokenizer:PreTrainedTokenizer,cutoff=None,eval_split=.1,replace_newline=True):

    fulltext = ''
    
    #Get text
    for item in os.listdir(ds_root): 
        item        = os.path.join(ds_root,item)
        full_text   = open(item,"r",encoding='utf-8').read()
        
        #Replace w no newline version
        if "\n" in full_text:
            full_text = full_text.replace("\n"," ")
            with open(item,"w",encoding='utf-8') as file:
                file.write(full_text)
            
        if replace_newline:
            full_text = full_text.replace("\n"," ")
        fulltext += full_text + "<|endoftext|>"

    #Pad 
    data        = [] 

    tokenized_text  = tokenizer(fulltext)['input_ids']

    while len(tokenized_text) > chunk_size:
        ids     = tokenized_text[:chunk_size]
        mask    = [1 for _ in range(chunk_size)]

        data.append({"input_ids":ids,"attention_mask":mask})
        tokenized_text  = tokenized_text[chunk_size:]
    
    random.shuffle(data)


    if not cutoff is None:
        data        = data[:cutoff]
        split_i     = int(len(data) * (1 - eval_split))
        train_data  = data[:split_i]
        test_data   = data[split_i:]
        return train_data,test_data
    
    split_i     = int(len(data) * (1 - eval_split))
    return data[:split_i],data[split_i:] 

    def tokenize(text):
        return tokenizer(text,padding=False)
        
    padded_dataset  = list(map(tokenize,data))
    return padded_dataset  

def create_vocab(ds_root:str,
                 tokenizer:PreTrainedTokenizer,
                 vocab_size:int,
                 chunk_size:int
                 ):
    
    #Load all text
    tx  = time.time()
    corpus      = ""
    #Create Corpus
    for file in os.listdir(ds_root):
        filename = os.path.join(ds_root,file)

        #Add text to corpus, all lower case
        with open(filename,"r",encoding='utf-8') as file:
            text        = file.read().lower()
            if False and "\uf8f3" in text:
                input(f"found in {filename}")
            removes     = {"\u200a":" ","\u2006":" ","  ":" ","\u2005":" ","\uf005":",","\uf001":"fi","\u2002":" ","\uf8ec":"|","\u2008":" ","⁄":"/","\uf8f3":"c","\uf8f7":"c","\u2007":" ","\uf8f0":"-"}

            for bad,good in removes.items():
                text = text.replace(bad,good)

            corpus += text

    
    print(f"corpus created with len: {len(corpus)}")


    #Iterate until vocab size reached
    tokens          = [char for char in  corpus]
    unique_tokens   = set(tokens)
    #print(f"finished load in {(time.time()-tx):.3f}s\n\tunique:{len(unique_tokens)}:\n{unique_tokens}")
    print(f"{len(unique_tokens)} vs {vocab_size}")
    
    while len(unique_tokens) < vocab_size:
        tx = time.time()
        print(f"tokens size {len(unique_tokens)}\tsequence size {len(tokens)}",end='')

        #Generate stats 
        pairs       = {}
        top_pair    = ""
        top_count   = 0
        for i in range(len(tokens)-1):

            pair    = tokens[i] + tokens[i+1]

            if pair in pairs:
                pairs[pair] += 1
            else:
                pairs[pair] = 1 


        pairs   = [(k,v) for k,v in pairs.items()]
        pairs.sort(key=lambda x: x[1],reverse=True)
       
        #Go through top 50 pairs and get non_conflicting 
        used        = []
        replacers   = []
        for pair in pairs[:50]:
            done        = False
            pair = pair[0]
            #Verify and add all leters
            for l in pair:
                if l in used:
                    done = True
                    break
    
            if done:
                continue
            else:
                for l in pair:
                    used.append(l)
                #Add to replacers if good
                replacers.append(pair)

        #Perform replacement
        print(f" replacing {replacers}")
        newtokens   = []
        join_i      = 0
        was_pair    = False



        for i in range(len(tokens)-1):
            
            if was_pair:
                was_pair = False 
                continue 

            #Add first token to newtokens 
            curtoken    = tokens[i]
            newtokens.append(curtoken)

            #Check if skipping next token

            if curtoken + tokens[i+1] in replacers:
                newtokens[-1] += tokens[i+1]
                was_pair = True

                    #join_i = 8 
            
            # if join_i:
            #     if join_i == 1:
            #         print(f"tokens: {newtokens[-32:]}")
            #     join_i -= 1


            
        tokens          = newtokens
        unique_tokens   = set(tokens)
        print(f"\t{(time.time()-tx):.2f}s")

                
    import json 
    with open(f"vocabulary.txt","w",encoding='utf_16') as file:
        print(f"writing {unique_tokens}")
        file.write(json.dumps(list(unique_tokens)))
        
        
    
    #Generate vocab
    tokenizer   = GPTSteinsharkTokenizer(vocab_size)
    
      

class GPTDataSet(Dataset):

    def __init__(self,data):
        self.data   = data 
    
    def __getitem__(self,i):
        return self.data[i]
    
    def __len__(self):
        return len(self.data)
    
# Returns True if successfully executes and writes to a file
def get_links(in_file, out_file):

    try:
        with open(in_file, "r", encoding='utf-8') as file:
            wiki_info = file.read()
        pattern = re.compile(r'href="(\/wiki\/[^"]+)"') #re.compile(r'href="(\/wiki\/[^"]+|https?:\/\/[^"]+)"') #re.compile(r'<a\s+href="([^"]+)"')
        hrefs = pattern.findall(wiki_info)

        # If hrefs exists, write to file
        if hrefs:
            with open(out_file, "w", encoding="utf-8") as o_file:
                o_file.write('\n'.join(hrefs))
    except Exception as e:
        print("get_links function did not work because:\n", e)
        return False

    return True


def get_readmes():
    corpora     = ["abc","brown","gutenberg","inaugural","","","","","","movie_reviews","","","","","","","","","","","","","","","","shakespeare","","state_union","","","",
                   "","","","","","","","","","","","","","","webtext","","","","","","","","","","","",""]
    corpus.abc

    # #Save abc 
    # with open("C:/code/nlp/data/abc.txt","w",encoding='utf-8') as file:
    #     file.write(corpus.abc.raw().replace("\n\n","<|endoftext|>"))

    # with open("C:/code/nlp/data/brown.txt","w",encoding='utf-8') as file:
    #     file.write(corpus.abc.raw().replace("\n\n","<|endoftext|>"))


def parse_time(filename,top_k=10):
    lines   = open(filename,"r").readlines()

    for line in lines:
        pass 
if __name__ == "__main__":
    t0 = time.time()
    create_vocab("C:/code/nlp/data",None,32768,16)
    print(f"finish in {(time.time()-t0):.3f}s")