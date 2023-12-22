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
import requests
import subprocess 
import sys 
import typing
from string import ascii_lowercase,ascii_uppercase
import multiprocessing
from collections import defaultdict
import json 
import requests
import bs4

GOOD_CHARS          = ascii_lowercase + r".,'?{}[]/\;:!@#$%^&*()1234567890-_=+ |~<>©°•·×→" + '"' + ascii_uppercase + "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
#sys.path.append(os.curdir)


def load_corpus(ds_root:str,lower:bool=True,eot:bool=True,newline:bool=True):
    corpus      = ""
    #Create Corpus
    for file in os.listdir(ds_root):
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
    corpus  = load_corpus(ds_root)
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
    for token in tokens:
        corpus  = corpus.replace(mappings[token],token)

    #print(f"{corpus[:64]}")
    #Iteratate until done 
    while len(tokens) < vocab_size:

        print(f"n_tokens={len(tokens)}/{vocab_size}\tcorpus len={len(corpus)}",end='',flush=True)
        t0  = time.time()
        #Find pair stats
        pairs   = defaultdict(int)
        for i in range(len(corpus)-1):
            pairs[f"{corpus[i]}{corpus[i+1]}"] += 1
            #pair    = f"{corpus[i]}{corpus[i+1]}"
            #input(f"pair='{pair}'->{pairs[pair]}")
        
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
    parsed_texts    = []
    explicit_count  = 0
    total_count     = 0 

    #Grab off header 
    pri_header      = ''
    for _ in range(pri_header_len):
        pri_header  += file.readline()

    #Used to determine if good source
    greek           = "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
    good_chars      = ascii_lowercase + r".,'?{}[]/\;:!@#$%^&*()1234567890-_=+ |~<>©°•·×→" + '"' + ascii_uppercase + greek
    quality_cutoff  = .7

    #Parse each page
    while file:

        #Grab page header 
        headertext      = ''
        for _ in range(header_len):
            headertext     += file.readline()
        if not headertext:
            break
        #ID language 
        page_lang       = headertext.split("\n")[7].split(": ")[-1].rstrip()
        saving          =  page_lang in languages


        page_text       = '' 
        quality_count   = 0
        n_chars         = 0
        lines_read      = 0
        trash           = False
        while True:
            
            #Read line in 
            line        = file.readline()
            lines_read  += 1

            #If in languages, is > 50 in len, save to text
            if saving and len(line) > 50:
                
                #Do check for first 20
                if lines_read < 20:
                    for char in line[:200].lower():
                        if char in good_chars:
                            quality_count += 1
                    n_chars += len(line[:200])

                #Make determination after 20 
                else:
                    if quality_count/(n_chars+.001) < quality_cutoff:
                        trash   = True
                        saving  = False

                #replace all chars in page_text
                new_page_text = "" 
                for char in line:
                    if not char in good_chars:
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
                            #print(f"no ref for char {char}")
                            new_page_text += "?"
                    else:
                        new_page_text += char
                page_text += new_page_text

            if line     == '\n':
                #Confirm is endl, then break
                if file.readline() == '\n':
                    break 

        if saving:
            total_count += 1
            if page_text.count('fuck') > 4 or page_text.count('porn') > 4 or page_text.count('hardcore') > 5:
                explicit_count += 1
                pass 
            else:
                parsed_texts.append(page_text)
        page_text   = ''

    print(f"skipped {explicit_count}/{total_count}")
    return parsed_texts


'''
    DESCRIPTION:
        download_crawl pulls files (100MB of compressed, text-extracted
        websites each) from the common crawl, unzips it, parses it for 
        only english, and saves files of specified size to the ds_path.
        *** "datacollection/wet.paths" is an expected file for this function.

    PARAMETERS:
        crawl_size          [int]   :   final size of the crawl in MB of english-only text 
        ds_path             [str]   :   path to save all text files to 
        file_size           [int]   :   size in MB of each file to be saved 
        rand_selection      [bool]  :   determines if crawl ursl are random or sequential from 0
'''
def download_crawl(crawl_size:int,ds_path:str,file_size:int,rand_selection:bool):

    #Generate URLS
    with open("datacollection/wet.paths","r") as crawl_url_file:

        crawl_urls  = [f"https://data.commoncrawl.org/{url}".replace('\n','') for url in crawl_url_file.readlines()]

    #Randomize as specified    
    if rand_selection:
        random.shuffle(crawl_urls)
    
    #Settings
    end_token           = "<|ENDOFTEXT|>"
    current_size_MB     = 0
    total_size_MB       = 0
    current_file        = max([int(f.replace('.txt','')) for f in os.listdir('data')] + [0])
    writable_file       = open(f"{ds_path}/{current_file}.txt","w",encoding='utf_8')
    
    while True:

        #Grab and download next URL
        next_url        = crawl_urls.pop(0)
        filename        = f"datacollection/{next_url.split('/')[-1]}"

        #Download and unzip gunzip
        #subprocess.run(f"echo off")
        subprocess.run(f"curl {next_url} -o {filename}")
        subprocess.run(f'7z x {filename} "-o{filename.replace(".gz","")}"')

        #Parse for eng documents 
        filename        = filename.replace('.gz','') 
        filename        = f"{filename}/{filename.replace('datacollection/','')}"
        with open(filename,'r',encoding='utf_8') as file:
            parsed_texts = parse_wet_file(file,['eng'])
        
        #add all texts to the current file 
        for text in parsed_texts:
            text_addition   = text + end_token
            text_len        = len(text_addition.encode('utf-8'))

            writable_file.write(text_addition)

            current_size_MB += text_len/1_000_000
            total_size_MB   += text_len/1_000_000

            if current_size_MB > file_size:
                print(f"current file size [{current_size_MB:.2f}MB] > {file_size}. Writing file")
                writable_file.close()
                current_file        += 1
                current_size_MB     = 0 
                writable_file       = open(f"data/{current_file}.txt","w",encoding='utf_8')
            if total_size_MB > crawl_size:
                print(f"Crawl download complete: [{total_size_MB:.2f}MB]. exiting")
                writable_file.close()
                return 
        
        #Cleanup file 


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
            text    = bs4.BeautifulSoup(text,features="lxml").text
            with open(filename,'w',encoding='utf_8') as file:
                file.write(make_good(text))
            file.close()
            
        else:
            print(f"got {response.status_code} for {url}")
    
        time.sleep(random.randint(0,2))
    

def create_dataset(root='alldata'):
    
    if not os.path.exists(root):
        os.mkdir(root)

    for droot in ['curated_data','data']:
        for filename in os.listdir(droot):
            filename    = os.path.join(droot,filename)

            with open(filename,"r",encoding='utf_8') as file:
                contents    = file.read()
            with open(filename.replace(droot,root),"w",encoding='utf_8') as file:
                file.write(make_good(contents))
        




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
    download_crawl(256,"data",8,True)
    #create_vocab_threads("C:/code/nlp/data",1024,n_threads=12)
    download_wiki()
    #create_dataset()
    create_vocab_whole("C:/code/nlp/alldata",1024)
    #search("C:/code/nlp/alldata",'))))')