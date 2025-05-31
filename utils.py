from torch.utils.data import Dataset
import os 
from transformers import PreTrainedTokenizer
import random
import re
from nltk import corpus
import time 
import requests
import subprocess 
import sys 
import typing
from string import ascii_lowercase,ascii_uppercase
from collections import defaultdict
import json 
import requests
import bs4
import math 
import numpy 
from hashlib import sha256 
import xxhash
GOOD_CHARS          = ascii_lowercase + r".,'?{}[]/\;:!@#$%^&*()1234567890-_=+ |~<>©°•·×→" + '"' + ascii_uppercase + "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
#sys.path.append(os.curdir)

SEEN_URI            = "C:/data/nlp/urls/prev.json" if os.path.exists("C:/data/nlp/urls/prev.json") else ""
print(f"seen uri is {SEEN_URI}")
SEEN_TEXTS:set      = set(json.loads(open(SEEN_URI,'r').read()))

class WebPage:

    def __init__(self,filetext:typing.TextIO,length_cutoff=500):
        #Read off "WARC-Type: conversion"
        filetext.readline()
        #Read url 
        self.url            = filetext.readline().replace("WARC-Target-URI: ","")

        #Read Date 
        self.date           = filetext.readline().replace("WARC-Date: ","")
        filetext.readline()
        filetext.readline()
        filetext.readline()
        #Read language,type
        self.lan            = filetext.readline().strip().replace("WARC-Identified-Content-Language: ","")
        self.type           = filetext.readline().strip().replace("Content-Type: ","")
        #Get length
        try:
            self.length     = int(filetext.readline().strip().replace("Content-Length: ",""))
        except ValueError:
            #Assume the best
            self.length     = length_cutoff + 1

        #Save line lengths
        self.line_lengths   = []
        #Determine if skipping or reading 
        self.discovered_eof = False 
        if not (self.lan == "eng" and self.length > length_cutoff and self.type == "text/plain"):
            self.include    = False
            self.skip(filetext)
        else:
            self.include    = True
            self.parse(filetext)
    

    def skip(self,filetext:typing.TextIO):
        nextline        = filetext.readline()
        while not nextline == "WARC/1.0\n":
            nextline    = filetext.readline()
            if not nextline:
                break
        
        if not nextline:
            self.discovered_eof = True


    def parse(self,filetext:typing.TextIO):
        nextline        = filetext.readline()
        self.contents   = nextline
        self.line_lengths.append(len(nextline))
        while not nextline == "WARC/1.0\n":
            nextline    = filetext.readline()
            if not nextline:
                break
            #dont include lines less than 40 chars 
            if len(nextline) >= 40:
                self.contents += nextline
                self.line_lengths.append(len(nextline))
        
        if not nextline:
            self.discovered_eof = True

        self.contents   = self.contents.replace("WARC/1.0","")
        while "\n\n" in self.contents:
            self.contents = self.contents.replace("\n\n","\n")

        
        self.include    = self.checkpage()
        


    def checkpage(self):
        #check if its a coding page 
        checktext           = self.contents.lower()
        text_len            = len(checktext) 

        #if its less than 1500 words, its not good enough 
        if text_len < 1500:
            return False 

        #If average of line_lengths is < 200, no thanks 
        if sum(self.line_lengths) / len(self.line_lengths) < 200:
            return False 
        
        #coding_buzzwords    = ["python","tutorial","data", 'module', "dataset", "c++","cpp","g++","gpp","file","deep learning","numpy","neural network","network"]

        #lower it to make common ground
        # wordcounts          = sum(checktext.count(buzzword) for buzzword in coding_buzzwords)
        # if wordcounts / len(checktext.split(" ")) > .05:
        #     input(checktext)


        #Apparently a good number have this
        if "the previous page is sending you to" in checktext or "please confirm that you and not a robot" in checktext:
            return False 
        

        #If ratio of alphabet to fullsize is < .80 deny it
        alphabet_thresh = .80
        alphabet        = ascii_lowercase + "." + "?" + "!" + ","
        good_count      = sum([checktext.count(char) for char in alphabet])
        alphabet_ratio  = good_count / text_len 

        if alphabet_ratio < alphabet_thresh:
            return False
        

        #If contains adult content words, discard
        adult_triggers  = [" milf "," anal "," pussy "," cunt ","porn"," sex "," cum ", "fuck", "blowjob", "cock"] 
        adult_count     = sum([checktext.count(keyword) for keyword in adult_triggers])
        if (6*adult_count) / text_len > .005:    #Scale to account for word vs char
            return False 
          
        
        #If average word size is > 15, its not right 
        avg_size_max    = 11
        num_words       = len(checktext.split(" "))
        if text_len / num_words > avg_size_max:
            return False
        
        #If punctuation is less than .001, its a no go 
        comma_count     = checktext.count(",")
        period_count    = checktext.count(".")
        excl_count      = checktext.count("!")
        ques_count      = checktext.count("?")
        punct_count     = period_count+excl_count+ques_count+comma_count
        if punct_count / text_len < .002:
            return False
        

        #if comma ratio is over .1 its too much 
        dash_count      = checktext.count("-")
        if comma_count / num_words > .1:
            return False 
        
        #if comma count is above .05 and punct count is below .01
        if (comma_count / num_words > .075) and ((punct_count-comma_count) / text_len < .005):
            return False

        #if dash count is over .01 its too much 
        if dash_count / text_len > .01:
            return False
        
        #Do split check on legitimate punctuation and remove all items longer than 100 words 
        plausible_splits    = [".",'?',"!","\n",";"]
        old_text            = checktext
        for splitter in plausible_splits:
            checktext       = checktext.replace(splitter,"<|SPLIT|>")
        checktext           = checktext.split("<|SPLIT|>")
        bad_runons          = [textitem for textitem in checktext if len(textitem.split(" ")) > 100]
        
        bad_indices         = [(old_text.find(runon),old_text.find(runon)+len(runon)) for runon in bad_runons]
        bad_indices         = [item for item in bad_indices if not item[0] == -1]
        
        #do a dummy check
        for start,end in bad_indices:
            find_chunk      = self.contents[start:end]
            self.contents  = self.contents.replace(find_chunk,"")


        #Check for tell-tale sales shop quotes 
        telltales           = ['free shipping', "sold out"]
        
        
        #print(checktext)
        return True 


def reduce_arr(arr:list,newlen:int):
    if not arr:
        return []
    
    newlen      = max(1,newlen)
    arr         = numpy.asarray(arr)
    factor      = len(arr) // newlen
    reduced     = arr[-newlen*factor:].reshape(newlen, factor).mean(axis=1)
   
    return reduced


def load_corpus(ds_root:str,rand_select:float=1,lower:bool=True,eot:bool=True,newline:bool=True):
    corpus      = ""
    #Create Corpus
    file_list   = os.listdir(ds_root)
    if not rand_select == 1.0:
        random.shuffle(file_list)
        file_list   = file_list[:int(rand_select*len(file_list))] 

    for file in file_list:
        filename = os.path.join(ds_root,file)

        #Add text to corpus, all lower case
        with open(filename,"r",encoding='utf-8') as file:
            text        = file.read()
            if lower:
                text    = text.lower()
            if eot:
                text    = text.replace('<|endoftext|>'," ").replace('<|ENDOFTEXT|>'," ")
            if newline:
                text    = text.replace('\n',' ')
            corpus += text
    return corpus


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


def get_stats(token_list:str):
    pairs       = {}
    for i in range(len(token_list)-1):

        pair    = token_list[i] + token_list[i+1]

        if pair in pairs:
            pairs[pair] += 1
        else:
            pairs[pair] = 1 
    return pairs


def make_good(text:str):
    new_page_text     = ''
    for char in text:
        if not char in GOOD_CHARS:
            if char == '–' or char == '—':
                new_page_text += "-"
                continue
            elif char == '’' or char == 'ˈ' or char == '`':
                new_page_text += "'"
                continue
            elif char == '”':
                new_page_text += '"'
                continue
            elif char == '“':
                new_page_text += '"'
                continue
            elif char == '…':
                new_page_text += "..."
                continue

            elif ord(char) == 10:
                new_page_text += '\n'
            else:
                new_page_text += "?"
        else:
            new_page_text += char
    
    while "\n\n" in new_page_text:
        new_page_text   = new_page_text.replace('\n\n','\n')

    return new_page_text


def replace(arguments):
    tokens_list, replacers  = arguments
    newtokens_list          = []
    was_pair                = False
    for i in range(len(tokens_list)-1):
            
        if was_pair:
            was_pair = False 
            continue 

        #Add first token to newtokens 
        curtoken    = tokens_list[i]
        newtokens_list.append(curtoken)

        #Check if skipping next token

        if curtoken + tokens_list[i+1] in replacers:
            newtokens_list[-1] += tokens_list[i+1]
            was_pair = True

    return newtokens_list
    

def expand(token,mappings):
    pair    = mappings[token]
    if len(pair) == 1:
        return pair 
    else:
        return f"{expand(pair[0],mappings)}{expand(pair[1],mappings)}"


def create_vocab_OLD(ds_root:str,vocab_size:int):
    
    #Load all text
    tx  = time.time()
    corpus      = ""
    #Create Corpus
    for file in os.listdir(ds_root):
        filename = os.path.join(ds_root,file)

        #Add text to corpus, all lower case
        with open(filename,"r",encoding='utf-8') as file:
            text        = file.read().lower()
            corpus += text



    #Create list of free unicode characters
    cur_corpus      = corpus
    avail_tokens    = [chr(i) for i in range(32768)]
    mappings        = dict()
    mapped          = dict()
    tokens          = set()

    #Create initial token set
    for char in corpus:
        if char in mapped:
            pass 
        else:
            token               = avail_tokens.pop(0)
            mappings[token]     = char 
            mapped[char]        = token
            tokens.add(token)

    #Replace text with tokens 
    for token in tokens:
        cur_corpus  = cur_corpus.replace(mappings[token],token)

    #Iterate until 'vocab_size' tokens created 
    while len(set(cur_corpus)) < vocab_size:

        #print for stats
        tx = time.time()
        print(f"n_tokens={len(set(cur_corpus))}/{vocab_size}\tlen={len(cur_corpus)}",end='',flush=True)

        #Generate stats on each pair
        pairs = defaultdict()

        for i in range(len(cur_corpus)-1):
            pairs[f"{cur_corpus[i]}{cur_corpus[i+1]}"] += 1

        #Find top pair
        t12     = time.time()
        top_pair    = ''
        top_n       = 0 
        for k,v in pairs.items():
            if v > top_n:
                top_n       = v 
                top_pair    = k 
        
        #print for stats
        print(f"\tstat_t={(t12-tx):.2f}s\tsort_t={(time.time()-t12):.2f}s",end='',flush=True)
         

        #Create new token 
        token               = avail_tokens.pop(0)
        mappings[token]     = top_pair 
        mapped[top_pair]    = token
        tokens.add(token)

        #Replace pairs with token
        cur_corpus  = cur_corpus.replace(top_pair,token)
        print(f"\ttotal_t={(time.time()-tx):.2f}s",flush=True)

                

    print(f"created {len(set(cur_corpus))}tokens")
    final_tokens    = [expand(token,mappings) for token in set(cur_corpus)]
    print(f"created: {final_tokens}")
    with open('vocabulary.txt','w',encoding='utf-8') as file:
        file.write(json.dumps(final_tokens))
        
        
def create_vocab_perword(ds_root:str,vocab_size:int):
    
    #Load all text
    tx  = time.time()
    corpus      = load_corpus(ds_root)

    #Split corpus into words
    splits      = [f"{word} " for word in corpus.split(" ")]

    #Get individual words and frequencies
    wordcounts  = {} 
    for word in splits:
        if word in wordcounts:
            wordcounts[word] = [word,wordcounts[word][1]+1] 
        else:
            wordcounts[word] = [word,1] 

    display     = f"{[f'{w}:{wordcounts[w]}' for w in list(wordcounts.keys())[:10]]}"
    input(f"{display} - wordcounts(pre)")

    #Create list of free unicode characters
    avail_tokens    = [chr(i) for i in range(32768)]
    mappings        = dict()
    mapped          = dict()
    tokens          = set()
    
    words           = "s"

    #Create initial token set
    for char in corpus:
        if char in mapped:
            continue 
        else:
            token               = avail_tokens.pop(0)
            mappings[token]     = char 
            mapped[char]        = token
            tokens.add(token)

    #Replace with tokens 
    for word in wordcounts:
        inplace_word    = wordcounts[word][0]
        inplace_count   = wordcounts[word][1]
        for char in word:
            inplace_word    = inplace_word.replace(char,mapped[char])
        wordcounts[word]    = [inplace_word,inplace_count]


    display     = f"{[f'{w}:{wordcounts[w]}' for w in list(wordcounts.keys())[:10]]}"
    input(f"{display} - wordcounts(post)")


        # for word in wordcounts:
        #     for i in range(len(words)):
        #         words[i] = [words[i][0].replace(mappings[token],token),words[i][1]]
    
    #Iterate until 'vocab_size' tokens created 
    while len(tokens) < vocab_size:

        #print for stats
        tx = time.time()
        print(f"n_tokens={len(tokens)}/{vocab_size}",end='',flush=True)

        #Generate stats on each pair
        pairs       = {}
        for key in wordcounts:
            word    = wordcounts[key][0]
            wcount  = wordcounts[key][1]

            #If done, no need 
            if len(word) == 1:
                continue 
            
            #Check pairs in word
            for j in range(len(word)-1):
                pair    = f"{word[j]}{word[j+1]}"
                
                if pair in pairs:
                    pairs[pair] += wcount
                else:
                    pairs[pair] = wcount

        #Find top pair
        t12     = time.time()
        top_pair    = ''
        top_n       = 0 
        for k,v in pairs.items():
            if v > top_n:
                top_n       = v 
                top_pair    = k 
         

        #Create new token 
        token               = avail_tokens.pop(0)
        mappings[token]     = top_pair 
        mapped[top_pair]    = token
        tokens.add(token)

        #print for stats
        expanded    = expand(token,mappings).replace("\n",'/n')
        print(f"\tpair= '{expanded}'\tpair_n={top_n}\tstat_t={(t12-tx):.2f}s\tsort_t={(time.time()-t12):.2f}s",end='',flush=True)

        #Replace pairs with token
        for word in wordcounts:
            wordcounts[word]    = [wordcounts[word][0].replace(top_pair,mapped[top_pair]),wordcounts[word][1]]

        print(f"\ttotal_t={(time.time()-tx):.2f}s",flush=True)

                

    print(f"created {len(tokens)}tokens")
    corpus          = ''.join([w[0] for w in words])
    final_tokens    = [expand(token,mappings) for token in set(corpus)]
    print(f"created: {final_tokens}")
    with open('vocabulary.txt','w',encoding='utf-8') as file:
        file.write(json.dumps(final_tokens))
 

def create_vocab_whole(ds_root:str,vocab_size:int):
    
    #Create string of corpus
    corpus  = load_corpus(ds_root,rand_select=1,newline=False)
    #print(f"corpus is {corpus[:64]}")
    #Create tokenization mechanisms
    offset          = 256
    avail_tokens    = [chr(i+offset) for i in range(32768-offset)]
    mappings        = dict()
    mapped          = dict()
    tokens          = dict()

    #Tokenize corpus
    for char in corpus:
        if not char in mapped:
            next_token              = avail_tokens.pop(0)
            mapped[char]            = next_token
            mappings[next_token]    = char 
            tokens[next_token]      = True
            #print(f"{char}->{next_token}")
    #print(f"tokens are {tokens}")
    #Replace chars with tokens
    
    #Take only first, middle, and last 50M tokens 
    corpus  = corpus[:10_000_000] + corpus[int(len(corpus)/2)-10_000_000:int(len(corpus)/2)+10_000_000] + corpus[-10_000_000:]

    for token in tokens:
        corpus  = corpus.replace(mappings[token],token)

    #print(f"{corpus[:64]}")
    #Iteratate until done 
    while len(tokens) < vocab_size:

        print(f"n_tokens={len(tokens)}/{vocab_size}\tcorpus len={len(corpus)}",end='',flush=True)
        t0  = time.time()
        #Find pair stats
        pairs   = defaultdict(int)


        # def add_i(i:int):
        #     pairs[f"{corpus[i]}{corpus[i+1]}"] += 1

        # [add_i(i) for i in range(len(corpus)-1)]
        
        for i in range(len(corpus)-1):
            pairs[f"{corpus[i]}{corpus[i+1]}"] += 1

        #Find top pair 
        top_pair    = ''
        top_n       = 0  
        for pair,count in pairs.items():
            if count > top_n:
                top_pair    = pair 
                top_n       = count 
        t1  = time.time()
        #Tokenize top pair 
        next_token              = avail_tokens.pop(0)
        mappings[next_token]    = top_pair 
        mapped[top_pair]        = next_token
        tokens[next_token]      = True

        expanded    = expand(next_token,mappings).replace("\n",'/n')
        print(f"\tpair= '{expanded}'\tpair_n={top_n}\tstat_t={(t1-t0):.2f}s\tsort_t={(time.time()-t1):.2f}s",end='',flush=True)
        
        corpus  = corpus.replace(top_pair,next_token)

        print(f"\ttotal_t={(time.time()-t0):.2f}s",flush=True)


    #Save to file 
    with open('vocab.txt','w',encoding='utf_8') as file:
        file.write(json.dumps(list([expand(t,mappings) for t in tokens.keys()])))
    file.close()


def search(ds_root:str,searchword:str):
    corpus      = ""
    #Create Corpus
    for file in os.listdir(ds_root):
        filename = os.path.join(ds_root,file)

        #Add text to corpus, all lower case
        with open(filename,"r",encoding='utf-8') as file:
            text        = file.read().lower()
            corpus += text
    
    print(f"found term: {corpus.count(searchword)} times")


def find_ml(ds_root:str,whitelist:list[str]):

    for subdir in os.listdir(ds_root):
        subdir  = os.path.join(ds_root,subdir)
        


'''
    DESCRIPTION:
        jennyblacklist sees if the paragraph is a good paragraph or not for training

    PARAMETERS:
        filetext [str] : string of the paragraph to check 
        
    RETURNS:
        [bool] : True if the paragraph is good, False if the paragraph is bad
'''
def jennyblacklist(filetext:str):
    # Check for URLs (http://, https://, www, etc.)
    url_pattern = r'http[s]?://[^\s]+|www\.[^\s]+'
    if re.search(url_pattern, filetext):
        return False

    # Check for code-like structures (code blocks, hashes, bullets, etc.)
    code_pattern = r'```|<code>.*</code>|#|\*|\-|\d+\.'  # Matches hashes, bullet points, numbered lists
    random_string_pattern = r'[^\w\s]'  # Matches non-alphanumeric characters like random punctuation

    # If any code-like structures or random strings are found, return False
    if re.search(code_pattern, filetext) or re.search(random_string_pattern, filetext):
        return False

    # Ensure the paragraph contains actual words (not random gibberish)
    word_count = len(re.findall(r'\w+', filetext))
    if word_count < 20:  # If fewer than 20 words, likely not coherent or helpful
        return False
    
    # Check if the paragraph forms a recognizable sentence (capitalized start, punctuation at the end)
    if not re.match(r'^[A-Z].*[.!?]$', filetext.strip()):
        return False
    
    # Check if more than 5 lines in a row end in a newline, if so, then likely not a good paragraph
    if re.search(r'(.*\n){5,}', filetext):
        return False

    return True


'''
    DESCRIPTION:
        given a filetext in the format of a str, return a bool of whether or 
        not to include it in the training set. True -> include it 

    PARAMETERS:
        filetext            str :   text to consider
'''
def blacklist(filetext:str) -> bool:

    #lower it to make parsing and such easier
    filetext        = filetext.lower()

    #If ratio of alphabet to fullsize is < .50 deny it
    alphabet_thresh = .5
    alphabet        = ascii_lowercase + "." + "?" + "!"
    text_len        = len(filetext) 
    good_count      = sum([filetext.count(char) for char in alphabet])
    alphabet_ratio  = good_count / text_len 

    if alphabet_ratio < alphabet_thresh:
        return False
    
    #If contains adult content words, discard
    adult_triggers  = [" milf "," anal "," pussy "," cunt "] 
    adult_count     = sum([filetext.count(keyword) for keyword in adult_triggers])
    if (6*adult_count) / text_len > .01:    #Scale to account for word vs char
        print(f"denying\n{filetext}")
        input()
        return False   
    return 

'''
    DESCRIPTION:
        given a filetext in the format of a common crawl WET file, parse and return
        per specified requirements 

    PARAMETERS:
        filetext            [list(str)] :   full text of a WET file per line  
        langauges           [list(str)] :   languages to return 
'''
def parse_wet_file(file:typing.TextIO,languages:list[str])-> list[str]:

    pri_header_len  = 18 
    header_len      = 11
    dataset         = [] 
    explicit_count  = 0
    total_count     = 0 

    #Scrape off header
    for _ in range(pri_header_len):
        file.readline()

    #Parse each webpage

    #Read off the first 'WARC/1.0'
    file.readline()
    done    = False
    while file and not done:
        nextpage    = WebPage(file)
        if nextpage.include:
            dataset.append(nextpage)
        if nextpage.discovered_eof:
            done = True
    return dataset


'''
    DESCRIPTION:
        download_crawl_to_db pulls files (100MB of compressed, text-extracted
        websites each) from the common crawl, unzips it, and saves it to the 
        ds_path for further processing.


    PARAMETERS:
        crawl_size          [int]   :   final number of .wet files to download
        ds_path             [str]   :   path to save all text files to 
        rand_selection      [bool]  :   determines if crawl urls are random or sequential from 0
        path_to_urls        [str]   :   specifies a '\n' delimited file of urls to download
'''
def download_crawl_to_db(crawl_size:int,ds_path:str,rand_selection:bool,path_to_urls="C:/data/nlp/urls/wet.paths"):
    print(f"\n\nDownloading {crawl_size} files")
    #Establish paths required 
    #Create local download folder 
    dload_path_local    = "downloads"
    if not os.path.exists(dload_path_local):
        os.mkdir(dload_path_local)
    #Create ds_path 
    if not os.path.exists(ds_path):
        os.mkdir(ds_path)
    
    #WET file paths saved here
    path_to_wet     = path_to_urls

    #Check for already downloaded list (by index)
    pre_downloaded  = [fname.split(".wet")[0].replace(".txt","") for fname in os.listdir(ds_path)]

    #Generate URLS that we have not downloaded already
    un_downloaded   = [] 
    with open(path_to_wet,"r") as crawl_url_file:
        crawl_urls  = [url.strip() for url in crawl_url_file.readlines()]
        #crawl_idxs  = [url.split(".wet")[0] for url in crawl_urls]

        for url in crawl_urls:
            if not url.replace("/","").split(".wet")[0] in pre_downloaded:
                un_downloaded.append(url)

    
    if rand_selection:
        random.shuffle(un_downloaded)
    

    #Save the first 'crawl_size' files to db 
    for next_url in un_downloaded[:crawl_size]:
        #Get full save path 
        savepath        = os.path.join(ds_path,next_url.replace("/",""))

        #Update url 
        next_url        = "https://data.commoncrawl.org/" + next_url
        #Download and unzip gunzip
        #subprocess.run(f"echo off")
        subprocess.run(f"curl {next_url} -o{savepath}") #Take out path name
        subprocess.run(f'7z x "{savepath}" "-o{ds_path}"')
        #Remove old file 
        os.remove(savepath)

'''
    DESCRIPTION:
        generate_ds parses files found in ds_path, performs a selection test 
         and saves files of specified size to the ds_path.

    PARAMETERS:
        ds_size             [int]   :   final size of the ds in MB of english-only text 
        data_path           [str]   :   path to raw downloaded text files
        ds_path             [str]   :   path to store final ds in   
        file_size           [int]   :   size in MB of each file to be saved 
'''
def generate_ds(ds_size:int,data_path:str,ds_path:str,file_size:int):
    #Get paths of the raw dataset files 
    text_file_paths = [os.path.join(data_path,fname) for fname in os.listdir(data_path) if not ".gz" in fname]

    #Settings
    end_token           = "<|endoftext|>"
    current_size_MB     = 0
    total_size_MB       = 0
    current_file        = os.path.join(ds_path,f"{random.randint(100_000_000,999_999_999)}.txt")
    writable_file       = open(current_file,"w",encoding='utf_8')
    
    for fpath in text_file_paths:
        
        #Parse for eng documents 
        # try:
            with open(fpath,'r',encoding='utf_8') as file:
                parsed_texts:list[WebPage] = parse_wet_file(file,['eng'])

            #add all texts to the current file 
            for webpage in parsed_texts:
                text_addition   = webpage.contents + end_token
                text_len        = len(text_addition.encode('utf-8'))

                hashed          = xxhash.xxh3_64(text_addition.encode()).hexdigest()

                if hashed in SEEN_TEXTS:
                    continue 
                else:
                    writable_file.write(text_addition)
                    #add writen file to already seen content 
                    SEEN_TEXTS.add(hashed)

                current_size_MB += text_len/(1024*1024)
                total_size_MB   += text_len/(1024*1024)

                if current_size_MB > file_size:
                    print(f"current file size [{current_size_MB:.2f}MB] > {file_size}. Writing file")
                    writable_file.close()
                    current_size_MB     = 0 
                    current_file        = os.path.join(ds_path,f"{random.randint(100_000_000,999_999_999)}.txt")
                    writable_file       = open(current_file,"w",encoding='utf_8')   
                if total_size_MB > ds_size:
                    print(f"Crawl download complete: [{total_size_MB:.2f}MB]. exiting")
                    writable_file.close()
                    with open(SEEN_URI,'w') as writefile:
                        writefile.write(json.dumps(list(SEEN_TEXTS)))
                    return 
        # except:
        #     pass
        #Cleanup file 
    with open(SEEN_URI,'w') as writefile:
        writefile.write(json.dumps(list(SEEN_TEXTS)))


def download_wiki():

    urls    = open(f"curated_data/urls.txt").readlines()
    
    for url in urls:
        
        url         = url.rstrip()
        filename    = "curated_data/wiki_"+url.split('/')[-1]+".txt"
        if os.path.exists(filename):
            continue
        response    = requests.get(url,timeout=1)
        
        if response.status_code == 200:
            text    = response.text 
            text    = bs4.BeautifulSoup(text).text
            with open(filename,'w',encoding='utf_8') as file:
                file.write(make_good(text))
            file.close()
            
        else:
            print(f"got {response.status_code} for {url}")
    
        time.sleep(random.randint(0,2))
    

def create_dataset(root='alldata'):
    
    if not os.path.exists(root):
        os.mkdir(root)

    for droot in ['pydata','curated_data','data']:
        for filename in os.listdir(droot):
            filename    = os.path.join(droot,filename)

            with open(filename,"r",encoding='utf_8') as file:
                contents    = file.read()
            with open(filename.replace(droot,root),"w",encoding='utf_8') as file:
                file.write(make_good(contents))
        


class GPTDataSet(Dataset):

    def __init__(self,ds_root:str):
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


def find_python_files(root_dir,limit=1000):
    python_files = []
    #whitelist   =   ['machine','learning','neural','network','pytorch','sigmoid','model','gpt','']
    
    # Walk through the directory tree
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file has a .py extension
            if filename.endswith('.py'):
                file_path = os.path.join(foldername, filename)
                found = True    # Hardcode in 
                # for substr in whitelist:
                #     if substr in file_path:
                #         found= True
                #         break
                
                # Build the full path to the Python file
                if found or random.random() < .01:
                    python_files.append(file_path)
            
            # if len(python_files) > limit:
            #     return  

    return python_files


def param_edit(parameter,method):
    method(parameter)

#Takes a crawl-data file and returns a list of urls 
def generate_urls(fpath:str):
    urls        = []
    crawlname   = fpath.split('/')[-1]
    base_path   = os.path.join(fpath,"segments")

    for id in os.listdir(base_path):
        root    = os.path.join(base_path,id)
        for wet in os.listdir(root):
            newroot     = os.path.join(root,wet)

            for maintype in os.listdir(newroot):
                urls.append(f'https://data.commoncrawl.org/crawl-data/{crawlname}/segments/{id}/{wet}/{maintype}')
    
    return urls



if __name__ == "__main__":
    # generate_urls("C:/data/nlp/urls/crawl-data/CC-MAIN-2025-18")

    # writefile = open('C:/data/nlp/urls/crawl2025.txt','w')
    # for url in generate_urls("C:/data/nlp/urls/crawl-data/CC-MAIN-2025-18"):
    #     writefile.write(url + "\n")
    
    # print(f"done")
    # exit()

    commands    = sys.argv[1]
    
    if commands == 'download':
        download_crawl_to_db(8,"C:\\data\\nlp\\download",True)
    elif commands == 'generate':
        generate_ds(1024*16,"C:/data/nlp/download","C:/data/nlp/crawl",16)