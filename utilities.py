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
from collections import defaultdict, Counter
import json 
import requests
import bs4
import math 
import numpy 
from hashlib import sha256 
import xxhash
import unicodedata
import training 

SEEN_URI            = "D:/nlp/seen_texts.txt"
GOOD_CHARS          = ascii_lowercase + r".,'?{}[]/\;:!@#$%^&*()1234567890-_=+ |~<>©°•·×→" + '"' + ascii_uppercase + "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
#sys.path.append(os.curdir)
BLACK_SUBSTR        = ["/cart","/checkout","/add-to-cart","/shop","/store","/product","/products","/pricing","/plans","/subscribe","/buy","/order","/deals","/offers","/discount","/coupon","/compare","/review","/reviews","/affiliate"]
BLACK_URL           = ["amazon","ebay","etsy","aliexpress","shopify","walmart","target","bestbuy","newegg","booking","expedia","tripadvisor","airbnb"]
CTA_TELLTALES       = ["start your","our skilled","our expert","reach out to us","for more information",'free shipping', "lorem ipsum", "click here to learn more", "limited time offer", "don't miss out on", "no reviews yet", "we are one of the", "buy now","add to cart","sign up","subscribe","get started","learn more","request a quote","contact sales","limited time","act now","order today","free trial", "we specialize in", "term and conditions"]

def domain_filter_no_text(url:str):

    #Find reddit
    if "://reddit.com" in url and "comments" in url:
        return True 
    
    #Find geeks on python
    if "geeksforgeeks.org/python" in url:
        return True 
    
    if "://python.org" in url:
        return True 
    
    if "quora.com" in url and "answer" in url:
        return True

    if "w3schools.com" in url:
        return True 
    
    if "en.wikipedia.org" in url:
        return True 
    
    if "stackexchange" in url:
        return True 
    
    if "://britannica.com" in url:
        return True 
    
    if "scholarpedia.org" in url and "article" in url:
        return True 
    
    if "://stanford.edu" in url or "://harvard.edu" in url or "://mit.edu" in url:
        return True
    
    if "://gutenberg.org" in url:
        return True 
    
    if "://nytimes.com" in url:
        return True 
    
    if "://history.com" in url:
        return True 
    
    return False

def stack_filter(text:str):
    checktext   = text.lower()
    return "python" in checktext


def shannon_entropy(s):
    freqs = Counter(s)
    return -sum((c/len(s)) * math.log2(c/len(s)) for c in freqs.values())

class WebPage:

    def __init__(self,filetext:typing.TextIO,seen_texts:set=set()):
        # lines = [filetext.readline() for _ in range (20)]
        # import pprint     
        line                = filetext.readline()
        header              = [line] 

        #Save line lengths
        self.line_lengths   = []

        #Determine if skipping or reading 
        self.discovered_eof = False 
        self.lan            = None 
        self.contents       = None
        self.include        = False
        self.hardfail       = False
        
        while not line == '\n':
            line = filetext.readline()
            header.append(line)
            if len(header) > 25:
                self.hardfail = True
                return
            
        url_line    = [l for l in header if "WARC-Target-URI: " in l][0]
        self.url    = url_line.replace("WARC-Target-URI: ",'').lower()
        #print(self.url)

        #If /product in the name, insta no 
        commerce_triggers   = BLACK_SUBSTR + BLACK_URL
        for trigger in commerce_triggers:
            if trigger in self.url:
                self.include    = False 
                self.skip(filetext)
                return

        #Get URL 
        hashed          = xxhash.xxh3_64(self.url.encode()).hexdigest()[:16]
        if seen_texts and hashed in seen_texts:
            self.include    = False 
            self.skip(filetext)
            return
        
        if seen_texts:
            seen_texts.add(hashed)

        #Filter by language 
        try:
            lang_line   = [l for l in header if "WARC-Identified-Content-Language: " in l][0]
            self.lan    = lang_line.replace("WARC-Identified-Content-Language: ",'').strip()
            if not self.lan == 'eng':
                self.include    = False 
                self.skip(filetext)
                return
            
        except IndexError:
            self.include = False 
            self.skip(filetext)
            return 
            

        try:
            type_line   = [l for l in header if "Content-Type: " in l][0]
            self.type    = type_line.replace("Content-Type: ",'')
        except IndexError:
            pass

        #Save line lengths
        self.line_lengths   = []

        #Determine if skipping or reading 
        self.discovered_eof = False 

        self.parse(filetext)
        self.include    = self.checkpage()

        if self.include:
            #Remove blank chars
            #self.contents       = ''.join(c for c in self.contents if unicodedata.category(c)[0] != "C")
            PUNCT_MAP = {
                "“": '"', "”": '"',
                "‘": "'", "’": "'",
                "–": "-", "—": "-",
                "‐":'-', "′":"'",
                "×":"x","©":"(c)"
            }
            for k,v in PUNCT_MAP.items():
                self.contents = self.contents.replace(k,v)
            good_chars          = ascii_lowercase + ascii_uppercase + "0123456789 " + "!@#$%^&*()[]{}_-+=~`'" + '":;<>,./?|\\\n\t'
            text_chars          = list(set(self.contents))
            for char in text_chars:
                if not char in good_chars:
                    self.include = False   
                    #print(f"bad char {char}")  
            self.contents = re.sub(r'\.(?=[A-Z])', '. ', self.contents)


    def skip(self,filetext:typing.TextIO):
        nextline        = filetext.readline()
        while not nextline == "WARC/1.0\n":
            nextline    = filetext.readline()
            if not nextline:
                self.discovered_eof = True
                self.include = False
                break
        

    def parse(self,filetext:typing.TextIO,stack=False):
        contents        = []
        min_len         = 100
        nextline        = filetext.readline()
        self.contents   = nextline
        self.line_lengths.append(len(nextline))

        while not nextline == "WARC/1.0\n":
            nextline    = filetext.readline()
            if not nextline:
                self.discovered_eof = True
                break

            #dont include lines less than 40 chars 
            if len(nextline) >= 2:
                contents.append(nextline)
                self.line_lengths.append(len(nextline))



        self.contents   = '\n'.join(contents)
        self.contents   = self.contents.replace("WARC/1.0","")
        # while "\n\n" in self.contents:
        #     self.contents = self.contents.replace("\n\n","\n")


        #Replace content paragraphs
        #Filter paragraphs
        paragraphs          = [l for l in self.contents.split('\n')]
        good_paragraphs     = [] 

        for i in range(len(paragraphs) - 1):
            next_i          = i + 1 
            skip            = False
            cur_par         = paragraphs[i]
            short_flag      = False
            saved_by_next   = False 
            #Only accept long content or possible headers/titles 
            if not len(cur_par) > min_len:
                short_flag = True 
                
                if len(paragraphs[next_i]) > min_len and len(cur_par) >= 1:
                    saved_by_next = True
                else:
                    continue


            #Skip paragraphs with not enough/too much punctuation
            comma_count     = cur_par.count(",")
            period_count    = cur_par.count(".")
            excl_count      = cur_par.count("!")
            ques_count      = cur_par.count("?")
            punct_count     = period_count+excl_count+ques_count+comma_count
            if punct_count / len(cur_par) < .005 and not saved_by_next:
                continue


            good_paragraphs.append(cur_par)            


        #Remove paragraphs that don't have enough punctuation
        self.contents       = '\n'.join(good_paragraphs)
        self.contents       = unicodedata.normalize("NFKC",self.contents)
        


    def checkpage(self):

        #Check to drop based on min length
        text_len            = len(self.contents) 
        if text_len < 1500:
            return False 
        
        #Perform comparisson on only lowercase
        checktext           = self.contents.lower()

        #Immediate negative for key phrases
        if "the previous page is sending you to" in checktext or "please confirm that you and not a robot" in checktext:
            return False 
        #Adult
        adult_triggers  = [" milf "," anal "," pussy "," cunt ","porn"," sex "," cum ", "fuck", "blowjob", "cock", "casino","jackpot","slots","free spin"] 
        adult_count     = sum([checktext.count(keyword) for keyword in adult_triggers])
        if (6*adult_count) / text_len > .005:    #Scale to account for word vs char
            return False 
        #Commerce
        total_counts    = 0
        for phrase in CTA_TELLTALES:
            total_counts += checktext.count(phrase)

            if total_counts >= 1:
                return False

        #If ratio of alphabet to fullsize is < .95
        alphabet_thresh = .95
        alphabet        = ascii_lowercase + "." + "?" + "!" + ","  + "'" + " " + "0123456789"
        alphabet_ratio  = sum([checktext.count(char) for char in alphabet]) / text_len 

        if alphabet_ratio < alphabet_thresh:
            return False
        
        #If average word size is > 15, its not right 
        avg_size_max    = 12
        num_words       = len(checktext.split(" "))
        if text_len / num_words > avg_size_max:
            return False
        
        #If punctuation is less than .002, its a no go 
        comma_count     = checktext.count(",")
        period_count    = checktext.count(".")
        excl_count      = checktext.count("!")
        ques_count      = checktext.count("?")
        punct_count     = period_count+excl_count+ques_count+comma_count
        if punct_count / text_len < .002:
            return False
        
        #if comma ratio is over .1 its too much 
        if comma_count / num_words > .1:
            return False 
        
        #if comma count is above .075 and punct count is below .005
        if (comma_count / num_words > .075) and ((punct_count-comma_count) / text_len < .005):
            return False

        #if dash count is over .01 its too much 
        dash_count      = checktext.count("-")
        if dash_count / text_len > .01:
            return False
        
        
        #Do split check on legitimate punctuation and remove all items longer than 100 words 
        plausible_splits    = [".",'?',"!"]
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
        
        #print(checktext)
        #input
        #(self.contents)
        return True


    #Applies standard formatting to accepted texts
    def format_content(self):



        #Filter paragraphs
        paragraphs          = [l for l in self.contents.split('\n')]
        good_paragraphs     = [] 

        for i in range(len(paragraphs) - 1):
            next_i          = i + 1 
            skip            = False
            cur_par         = paragraphs[i]
            short_flag      = False
            saved_by_next   = False 
            #Only accept long content or possible headers/titles 
            if not len(cur_par) > 100:
                short_flag = True 
                if len(paragraphs[next_i]) > 100 and len(cur_par) >= 1:
                    saved_by_next = True
                else:
                    continue


            good_paragraphs.append(cur_par)            


        #Remove paragraphs that don't have enough punctuation
        self.contents       = '\n'.join(good_paragraphs)
        
        
        #if all conttents < 1000 chars, forget it 
        if len(self.contents) < 1000:
            return False

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
def parse_wet_file(file:typing.TextIO,seen_texts:set=set())-> list[WebPage]:
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
    total_pages     = 0
    total_eng       = 0 
    while file and not done:
        nextpage        = WebPage(file,seen_texts=seen_texts)

        if nextpage.hardfail:
            print(f"using {len(dataset)} /{total_eng} ENG /{total_pages}")
            return dataset
        if nextpage.include:
            dataset.append(nextpage)
            #input(f"\n\nACCEPTED:\n\n{nextpage.contents}")
        if nextpage.discovered_eof:
            done = True
        if nextpage.lan == 'eng':
            total_eng += 1
        total_pages += 1
    print(f"using {len(dataset)} /{total_eng} ENG /{total_pages}")
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
def download_crawl_to_db(crawl_size:int,ds_path:str,rand_selection:bool,path_to_urls="D:/nlp/temp/wet.paths"):
    print(f"\n\nDownloading {crawl_size} files")
    #Establish paths required 
    #Create local download folder 
    dload_path_local    = "downloads"
    downloaded          = 0 
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
        subprocess.run(f"curl {next_url} -o{savepath}",check=True) #Take out path name
        subprocess.run(f'7z x "{savepath}" "-o{ds_path}"',check=True)
        #Remove old file 
        os.remove(savepath)
        parse_archive_file(savepath.replace(".gz",""))

        downloaded += 1 
        print(f"\n\n{downloaded}/{crawl_size}\n\n")

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
    text_file_paths     = [os.path.join(data_path,fname) for fname in os.listdir(data_path) if not ".gz" in fname and not ".wtc" in fname and ".wet" in fname]
    seen_texts          = open(SEEN_URI,'r').read()
    SEEN_TEXTS          = set(json.loads(seen_texts if seen_texts else "[]"))
    #Settings
    end_token           = training.END_TOKEN
    current_size_MB     = 0
    total_size_MB       = 0
    current_file        = os.path.join(ds_path,f"{random.randint(100_000_000,999_999_999)}.txt")
    writable_file       = open(current_file,"w",encoding='utf_8')
    
    for fpath in text_file_paths:
        
        #Parse for eng documents 
        # try:
            with open(fpath,'r',encoding='utf_8') as file:
                parsed_texts:list[WebPage] = parse_wet_file(file,['eng'],filter_domain=False,seen_texts=SEEN_TEXTS)

            #add all texts to the current file 
            for webpage in parsed_texts:
                text_addition   = webpage.contents + end_token
                text_len        = len(text_addition.encode('utf-8'))

                writable_file.write(text_addition)
                #add writen file to already seen content 

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
            if random.random() < .02:
                print(f"Writing seen")
                with open(SEEN_URI,'w') as writefile:
                    writefile.write(json.dumps(list(SEEN_TEXTS)))

        # except:
        #     pass
        #Cleanup file 
    with open(SEEN_URI,'w') as writefile:
        writefile.write(json.dumps(list(SEEN_TEXTS)))


#Parses a .wet file inplace for general cleaning
def parse_archive_file(fpath:str):
    
    newfname    = fpath.replace(".wet",".wetc")
    if os.path.exists(newfname):
        print(f"existed {newfname}")
        os.remove(fpath)
        return 
    
    print(f"rework {fpath}")
    with open(fpath,'r',encoding='utf_8') as file:
        #Grab all contents
        documents   = parse_wet_file(file,None)

    file.close()
    time.sleep(.2)
    os.remove(fpath)

    with open(fpath.replace(".wet",".wetc"),'w',encoding='utf_8') as writefile:
        writefile.write(f"{training.END_TOKEN}".join([doc.contents for doc in documents]) + f"{training.END_TOKEN}")
    

def collect_docs(data_path:str):

    doc_db              = open(f"D:/nlp/crawl docs/db.txt",'w',encoding='utf-8')
    docs                = {}
    doc_i               = 0 

    text_file_paths     = [os.path.join(data_path,fname) for fname in os.listdir(data_path) if not ".gz" in fname and not ".wetc" in fname and ".wet" in fname]

    for fpath in text_file_paths:
        docs            = open(fpath,'r',encoding='utf-8').read().split(training.END_TOKEN)

        for doc in docs:
            docs[doc_i] = doc 
            doc_i       += 1

    doc_db.write(json.dumps(docs))
    print(f"Wrote {doc_i} documents to db")

def rework_archive(data_path:str):

    text_file_paths     = [os.path.join(data_path,fname) for fname in os.listdir(data_path) if not ".gz" in fname and not ".wetc" in fname and ".wet" in fname]

    for fpath in text_file_paths:
        parse_archive_file(fpath)


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

    if len(sys.argv) > 2:
        n_download = int(sys.argv[2])
    else:
        n_download = 8
    
    if commands == 'download':
        download_crawl_to_db(n_download,ds_path="D:\\nlp\\crawl",rand_selection=True)
    elif commands == 'generate':
        generate_ds(1024*64,"D:/nlp/crawl","D:/nlp/dump",64)
    elif commands == 'clean':
        rework_archive("D:\\nlp\\crawl")