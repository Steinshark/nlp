from tokenizers.implementations import ByteLevelBPETokenizer
import multiprocessing.pool
import torch 
from torch.utils.data import Dataset, Sampler
import numpy
import os 
import random 
from xxhash import xxh3_64
import unidecode
import multiprocessing
import sys 
#sys.path.append("youtubeDB")
from youtubeDB.utils import filter_bad_content
import json 
import queue 
import threading
import time 
from collections import Counter
from language_utils import ALL_MISSPELLINGS, CHAR_CORRECTIONS, REMOVAL_CHAR, REMOVAL_THRESH
import language_utils
import re 


TOTAL_TOKENS                = 0 
TOTAL_REPLACEMENTS          = 0 
class InfSampler(Sampler):

    def __init__(self):

        pass 

    def __iter__(self):
        while True:
            yield 1

    def __len__(self):
        return float("inf") 


class TextFileDataset(Dataset):

    #Assumption is that all files in ds_root are pre_cleaned and utf_8 encodable
    def __init__(self,ds_root:str,n_positions:int,max_files:int=1_000_000,tokenize_with=None):

        #Get list of filenames
        self.filenames      = [os.path.join(ds_root,file) for file in os.listdir(ds_root)][:max_files]  

        #Load all content
        self.texts          = [open(fname,'r',encoding='utf_8').read() for fname in self.filenames]

        self.tokens         = [] 
        #Set class variables
        self.n_positions    = n_positions

        #Start in warmup mode
        self.warmup         = True
        self.train_i        = .05    

        self.tokenize_with  = tokenize_with
        #Perform cleaning operations
        

    def print_stats(self):
       print(f"Created dataset:\n\t\t{len(self.text.split(' '))/1_000_000:.2f}M words\n\t\t{len(self.text)/1_000_000:.2f}M characters\n\t\t{len(self.tokenized_text)/1_000_000:.2f}M tokens\n\t\t{len(set(self.text))} unique chars\n\n")


    def as_dict(self):
        return {"text":self.text[i] for i in range(len(self.text))}
        
    def tokenize(self,tokenizer):
        print(f"attempting tokenization of {len(self.texts)}")
        token_chunks    = [tokenizer.encode(text).ids for text in self.texts]
        print(f"created results, appending")
        for chunk in token_chunks:
            self.tokens += chunk
        self.tokens     = numpy.asarray(self.tokens)

        print(f"self.tokens is {self.tokens.shape}")


            

    #returns text_ids and attention_mask
    def __getitem__(self,i):
        
        #Pick random window 
        start_i                 = random.randint(0,int(len(self.text)*self.train_i)-self.n_positions)

        return self.tokenized_text[start_i:start_i+self.n_positions]


        
    def __len__(self):
        return self.size


    def clean_text(self):

        #All lower 
        #self.text           = self.text.lower()
        #Remove double newlines
        changes     = {
            '\xa3':'L',
            '\xa0':' ',
            '\u2019':"'",
            '\xb0':'degrees',
            '\u266a':'[music]',
            '\u2026':'...',
            '\U0001f60a':':)',
            "\x2018":"'",
            "\u2018":"'",
            "\xe9":"e",
            "\u20ac":'Euro',
            "\u2044":"/",
            "\xe8":'e',
            "\xf1":"n",
            "\u201c":'"',
            "\u201d":'"',
            "\xed":'i',
            "\xc5":'A',
            "\xf3":'o',
            "\u2014":'-',
         #   '\n':' ',
            "'":'',
            'é':'e',
            "♪":'[music]',
            '\xa0':' ',
            " uh ": " ",
            " i i ": " ",
            " the the ": " ",
            " um ": " ",
            " it's it's ": " it's"
        }

        for x,y in changes.items():
            self.text   = self.text.replace(x,y).lower()

    @staticmethod
    def cleantext(text):

        text = text.lower()
        changes     = {
            '\xa3':'L',
            '\xa0':' ',
            '\u2019':"'",
            '\xb0':'degrees',
            '\u266a':'[music]',
            '\u2026':'...',
            '\U0001f60a':':)',
            "\x2018":"'",
            "\u2018":"'",
            "\xe9":"e",
            "\u20ac":'Euro',
            "\u2044":"/",
            "\xe8":'e',
            "\xf1":"n",
            "\u201c":'"',
            "\u201d":'"',
            "\xed":'i',
            "\xc5":'A',
            "\xf3":'o',
            "\u2014":'-',
         #   '\n':' ',
            "'":'',
            'é':'e',
            "♪":'[music]',
            '\xa0':' ',
            " uh ": " ",
            " i i ": " ",
            " the the ": " ",
            " um ": " ",
            " it's it's ": " it's"
        }
        for x,y in changes.items():
            text   = text.replace(x,y)
        
        return text
    
    def create_tokenized(self):
        pass


    def get_iter(self,max_i=100_000_000):
        i   = 0 
        for item in self.texts.split(self.eos_token):
            yield item
        return



    def save_to_file(self,fname:str):
        with open(fname,'w',encoding='utf_8') as file:
            file.write(self.texts)
        return 


class TokenizedDataset2(Dataset):

    def __init__(self,tokens,n_positions):
        self.tokens         = tokens 
        self.n_positions    = n_positions 

        self.n_tokens       = len(self.tokens)
        self.warmup         = False

    

    def __getitem__(self, index)->dict[str,torch.Tensor]:
        #Pick random start 

        end_point           = len(self.tokens) - (self.n_positions+1) if not self.warmup else int(len(self.tokens)*.02)
        start_i             = random.randint(0,end_point)

        #start_i             = 0

        token_seq           = numpy.asarray(self.tokens[start_i:start_i+self.n_positions])
        token_seq           = torch.from_numpy(token_seq).type(torch.int16)
        target_seq          = numpy.asarray(self.tokens[start_i+1:start_i+self.n_positions+1])
        target_seq          = torch.from_numpy(target_seq).type(torch.int16)
        return {"input_ids":token_seq,"target_ids":target_seq}
    

    def sample(self,bs:int,n_tokens:int,device:torch.device,holder)->dict[str,torch.Tensor]:
        end_point           = len(self.tokens) - (self.n_positions+1) if not self.warmup else int(len(self.tokens)*.02)
        idxs                = [random.randint(0, end_point) for _ in range(bs)]

        token_seqs          = torch.tensor(numpy.array([self.tokens[idxs[i]:idxs[i]+n_tokens] for i in range(bs)])).to(device,non_blocking=True).long()
        target_seqs         = torch.tensor(numpy.array([self.tokens[idxs[i]+1:idxs[i]+n_tokens+1] for i in range(bs)])).to(device,non_blocking=True).long()

        return {"input_ids":token_seqs,"target_ids":target_seqs}




    def __len__(self):
        return len(self.tokens) // self.n_positions


class TokenizedDataset(Dataset):

    def __init__(self, tokens, n_positions):
        if isinstance(tokens, numpy.ndarray):
            tokens = torch.from_numpy(tokens)
        self.tokens = tokens.contiguous().to(torch.int16)  # Make sure it's contiguous for fast slicing
        self.n_positions = n_positions
        self.n_tokens = len(self.tokens)
        self.warmup = False

        #self.batch_input     = torch.empty((bs,n_tok),dtype=torch.long,device=device,pin_memory=False)
        #self.batch_target    = torch.empty((bs,n_tok),dtype=torch.long,device=device,pin_memory=False)


    def build_idxs(self,bs,n_tokens):
        end_point = len(self.tokens) - (n_tokens + 1)
        return  torch.randint(0, end_point, (bs,))

    def stack_indices(self,n_tokens,idxs):
        offsets         = idxs.unsqueeze(1) + torch.arange(n_tokens).unsqueeze(0)
        batch_input = self.tokens[offsets]
        batch_target = self.tokens[offsets + 1]
        #batch_input     = torch.stack([self.tokens[i : i + n_tokens] for i in idxs])
        #batch_target    = torch.stack([self.tokens[i + 1 : i + n_tokens + 1] for i in idxs])

        return batch_input,batch_target
    
    def sample(self, bs: int, n_tokens: int, device=None, pin_memory=False) -> dict[str, torch.Tensor]:

        idxs                                = self.build_idxs(bs,n_tokens)

        
        batch_input,batch_target           = self.stack_indices(n_tokens,idxs)
        

        #self.batch_input.copy_(bi,non_blocking=True)
        #self.batch_target.copy_(bt,non_blocking=True)

        return {
            "input_ids": batch_input.to(device).long(),
            "target_ids": batch_target.to(device).long(),
        }

    def __len__(self):
        return self.n_tokens // self.n_positions


class Prefetcher:

    def __init__(self,dataset:TokenizedDataset,bs:int,n_tok:int,device:torch.device,queue_size=3):
        self.dataset = dataset
        self.batch_size = bs
        self.n_tokens = n_tok
        self.device = device
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = True

        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()


    def _worker(self):
        while self.running:
            if not self.queue.full():
                try:
                    batch = self.dataset.sample(self.batch_size, self.n_tokens, self.device,pin_memory=True)
                    self.queue.put(batch)
                except Exception as e:
                    print(f"[Prefetcher] Error: {e}")
                    time.sleep(0.1)  # Prevent busy looping

    def get_batch(self):
        return self.queue.get()

    def stop(self):
        self.running = False
        self.thread.join()

import multiprocessing
from functools import partial


#For some reason, after training tokenizer, it doesnt save EOT token when finished
def load_tokenizer(f_root:str)->ByteLevelBPETokenizer:
    tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename=f"{f_root}/vocab.json",merges_filename=f"{f_root}/merges.txt")
    print(f"init tokenizer size {tokenizer.get_vocab_size()}")
    tokenizer.add_tokens(["<|endoftext|>"])
    print(f"loaded tokenizer size {tokenizer.get_vocab_size()}")
    return tokenizer


def read_and_tokenize(fname, fpath, tokenizer:ByteLevelBPETokenizer):
    global TOTAL_TOKENS
    filename        = os.path.join(fpath,f"{random.randint(1_000_000_000,999_000_000_000)}")
    with open(fname, 'r', encoding='utf_8') as f:
        text = f.read() + "<|endoftext|>"
        ids         =  tokenizer.encode(text).ids
    
    tokens          = numpy.asarray(ids,dtype=numpy.uint16)
    TOTAL_TOKENS    += tokens.__len__()
    numpy.save(filename,tokens)
    return tokens.__len__()


def create_token_file_parallel(ds_root, tokenizer, save_dir, chunk_size=25_000_000):
    from tqdm import tqdm
    import numpy as np
    import os

    os.makedirs(save_dir, exist_ok=True)
    n_token_files   = 0
    total_tokens    = 0

    filenames = [os.path.join(ds_root, f) for f in os.listdir(ds_root) if os.path.isfile(os.path.join(ds_root, f))]


    # Create multiprocessing pool
    pool = multiprocessing.Pool(processes=8)
    tokenize_func = partial(read_and_tokenize, tokenizer=tokenizer,fpath=save_dir)
    results             = pool.imap(tokenize_func, filenames, chunksize=8)
    for result in results:
        print(result)

    # for token_list in tqdm(token_batches, total=len(filenames), desc="Tokenizing"):
    #     token_buffer += token_list

    #     if len(token_buffer) >= chunk_size:
    #         np_arr = np.asarray(token_buffer[:chunk_size], dtype=np.int32)
    #         np.save(os.path.join(save_dir, f"{n_token_files}.npy"), np_arr)
    #         n_token_files += 1
    #         total_tokens += len(np_arr)
    #         token_buffer = token_buffer[chunk_size:]

    # # Final dump
    # if token_buffer:
    #     np_arr = np.asarray(token_buffer, dtype=np.int32)
    #     np.save(os.path.join(save_dir, f"{n_token_files}.npy"), np_arr)
    #     total_tokens += len(np_arr)

    print(f"[✓] Finished tokenizing {TOTAL_TOKENS:,} tokens")


#All files are assumed to be cleaned
def create_token_file(ds_root,tokenizer:ByteLevelBPETokenizer):

    #Gather all filenames
    print(f"loading fnames")
    filenames       = [os.path.join(ds_root,file) for file in os.listdir(ds_root)]  
    #random.shuffle(filenames)
    #Create list of all texts
    #texts           = [None] * len(filenames) 
    n_token_files   = 0 
    total_tokens    = 0
    print(f"loading text",flush=True)
    print(f"tokenizing texts",flush=True)
    tokens          = [] 
    for i,fname in enumerate(filenames):
        with open(fname,'r',encoding='utf_8') as readfile:
            #texts[i] = readfile.read() + "<|endoftext|>"
            text        = readfile.read() + "<|endoftext|>"


    # i = 0
    # while texts:
    #     text        = texts.pop()
        #i           += 1 
        tokens += tokenizer.encode(text).ids
        if len(tokens) > 10_000_000:
            np_arr:numpy.ndarray  = numpy.asarray(tokens).flatten()
            np_arr                = np_arr.astype(int)
            numpy.save(f"C:/data/nlp/tokens/{n_token_files}.npy",np_arr)
            n_token_files += 1
            total_tokens += len(tokens)
            tokens = []
            print(f"tokenized {n_token_files}")

    np_arr:numpy.ndarray  = numpy.asarray(tokens).flatten()
    np_arr.astype(int)
    numpy.save(f"C:/data/nlp/tokens/{n_token_files}.npy",np_arr)
    print(f"created token set {total_tokens}")


def train_tokenizer(vocab_size=32768,train_root="C:/data/nlp/train_dir",name='stein_tok'):
    print(f"Training {name} tokenizer size={vocab_size}")
    tokenizer               = ByteLevelBPETokenizer()
    tokenizer.train([os.path.join(train_root,fname) for fname in os.listdir(train_root)],vocab_size=vocab_size-1)
    tokenizer.add_tokens(["<|endoftext|>"])

    if not os.path.exists(f"C:/data/nlp/{name}"):
        os.mkdir(f'C:/data/nlp/{name}')
    tokenizer.save_model(f'C:/data/nlp/{name}')
    print(f"\tcomplete - saved as {name}")


def get_yt_captions(ytdump_file:str='ytdump.html'):
    import youtube_transcript_api
    from youtube_transcript_api import YouTubeTranscriptApi
    import time 


    #Parse file for links 
    with open(ytdump_file,'r',encoding='utf_8') as read_file:

        filetext        = read_file.read()
        splittext       = filetext.split("/watch?v=")
        yt_ids          = set([text.split('"')[0] for text in splittext][1:])
        cleaned_ids     = []
        for ytid in yt_ids:
            if '&' in ytid:
                ytid = ytid.split("&")[0]
            if "t=" in ytid:
                ytid = ytid.split("t=")[0]

            cleaned_ids.append(ytid)


        grabber         = YouTubeTranscriptApi()
        for id in cleaned_ids:
            if os.path.exists(f"yt_captions/{id}.txt"):
                print(f"already had {id}")
                continue
            try:
                video_text   = " ".join([l['text'] for l in grabber.get_transcript(id)])
                print(f"found {len(video_text.split(' '))} words")
                with open(f"yt_captions/{id}.txt",'w',encoding='utf_8') as write_file:
                    write_file.write(video_text)
                    write_file.close()
            except youtube_transcript_api._errors.NoTranscriptFound:
                pass
            except youtube_transcript_api._errors.TranscriptsDisabled:
                pass


def add_file_to_db(fpath:str,final_dir:str,rootdir:str,removal_tokens:list):
    
    
    with open(fpath,'r',encoding='utf_8') as readfile:
        contents    = readfile.read()
        if "yt_ascii" in rootdir:
            contents    = json.loads(contents)['transcript']
        #good        = contents["is_quality"]

        #Clean contents    
        clean_contents  = clean_individual_text(contents,removal_tokens)
        content_hash    = xxh3_64(clean_contents.encode()).hexdigest()
        fpath_save      = os.path.join(final_dir,content_hash+".txt")

    #Ensure no duplicate contents
    if not os.path.exists(fpath_save):
        #Ensure not saving empty file
        if not filter_bad_content(contents):
            return 0,0
        chars           = len(clean_contents)
        words           = len(clean_contents.split(" "))

        #Write contents 
        with open(fpath_save,'w',encoding='utf_8') as writefile:
            writefile.write(clean_contents+language_utils.EOT_STR)
        return chars, words 
    else:
        return 0,0
    
def correct_by_dict(text: str) -> str:
    return language_utils.PATTERN.sub(lambda x: language_utils.ALL_CORRECTIONS[x.group(0)], text)

def correct_to_ascii(text:str) -> str:
    return language_utils.ONLYASCII.sub(lambda x: "",text)

def parallel_substitution(texts: list[str], num_workers: int = None) -> list[list[str],multiprocessing.Pool]:
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(correct_by_dict, texts)
    
    return results, pool

#Provided a list of directories of text files,
# combine these into train_dir based on contents 
def prep_data_for_training(desired_sources:list[str],final_dir="C:/data/nlp/train_dir",eot_token="<|endoftext|>"):
    
    if not os.path.exists(final_dir):
        os.mkdir(final_dir)
        
    #Clear dataroot 
    print(f"Cleaning root")
    for file in [os.path.join(final_dir,fname) for fname in os.listdir(final_dir)]:
        os.remove(file)

    print(f"Calculating token statistics")
    token_counts            = {}
    total_tokens            = 0
    i                       = 0 
    fileset                 = [] 



    #Generate list of all paths to data
    for rootdir in desired_sources:
        for fname in os.listdir(rootdir):
            fpath_load      = os.path.join(rootdir,fname)
            fileset.append(fpath_load)

    #Clean and return all data
    print(f"finding data")
    all_paths               = [os.path.join(rootdir,fname) for fname in os.listdir(rootdir) for rootdir in desired_sources]
    divs                    = 20
    for i in range(divs):
        start_i             = i * (len(all_paths) // divs)
        end_i               = start_i + (len(all_paths)//divs)

        print(f'\tloading data [{i}/{divs}]')
        content_section     = [open(fpath,'r',encoding='utf_8').read() for fpath in all_paths[start_i:end_i]]
        print(f"\tcleaning data")
        clean_contents,pool = parallel_substitution(content_section,num_workers=16)
        
        #Count unique tokens 
        for cont in clean_contents:
            total_tokens += len(cont)
            for tok in set(cont):
                token_counts[tok] = 0
        pool.join()


    print(f"found {len(token_counts)} unique tokens")

    #Clean using 5% of data to make statistics for what to drop
    random.shuffle(fileset)
    content_selection       = [open(fpath,'r',encoding='utf_8').read() for fpath in fileset[:len(fileset)//20]]
    clean_contents,pool     = parallel_substitution(content_selection,num_workers=16)
    counts                  = [Counter(content) for content in clean_contents]
    for count in counts:
        token_counts.update(count)

    removal_tokens          = [tok[0] for tok in token_counts.items() if ((tok[1]/total_tokens) < REMOVAL_THRESH) and not (tok[0] in language_utils.GOOD_CHAR)]
    print(f"dataset contains:\t{len(token_counts)} unique tokens")
    print(f"dataset contains:\t{total_tokens} tokens")
    print(f"marked {len(removal_tokens)} tokens for removal from set")
    

    #stats 
    chars                   = 0 
    words                   = 0 

    with multiprocessing.Pool(12) as pool:
        results             = pool.starmap(add_file_to_db,[(p,final_dir,rootdir,removal_tokens) for p in all_paths])

    
    #results                 = map(add_file_to_db,all_paths,[final_dir for _ in all_paths],[rootdir for _ in all_paths],[removal_tokens for _ in all_paths])
    
    for result in results:
        c,w                 = result 
        chars += c 
        words += w 

    pool.join()
    texts                   = len(os.listdir(final_dir))

    print(f"gathered texts\n\tchars: {chars//1_000_000}M\n\twords: {words//1_000_000}M\n\ttexts: {texts}")
    return

def generate_clean_contents(fpath:str):
    contents            = open(fpath,'r',encoding='utf_8').read()
    corrections         = CHAR_CORRECTIONS.copy()
    corrections.update(ALL_MISSPELLINGS)
    clean_contents  = correct_by_dict(contents,corrections)

    return clean_contents


def remove_tokens(text:str, tokens_to_remove:list):
    return text.translate(str.maketrans('', '', ''.join(tokens_to_remove)))

#Create a ASCII version of the transcripts 
# also take out stops words and other stupid stuff
def clean_individual_text(contents:str,removal_tokens:list[str]):

    clean_contents  = correct_by_dict(contents)
    clean_contents  = remove_tokens(clean_contents,removal_tokens)
    #clean_contents  = clean_contents.lower()
    return clean_contents.strip()



def find_by_topic(final_dir="C:/data/nlp/train_dir",topic_keywords={"project":2,"application":5,"variable":6,"python":25,"c++":25,"cpp":25,"g++":25,"pytorch":25,"coding":25,"program":25,"parser":25,"computer":8,"neural network":25,"computer science":25,"comput":4,"byte":7,"tutorial":6,"code":6,"developer":12}):


    scores  = {} 
    root    = "C:/data/nlp/"
    for fpath in [os.path.join(final_dir,fname) for fname in os.listdir(final_dir)]:
        
        #Get contents score DENSITY 
        with open(fpath,'r') as readfile:
            contents    = readfile.read()
            file_score  = 0 
            indv_score  = [] 
            #Get score 
            for topic,score in topic_keywords.items():
                indv_score.append((contents.count(topic)*score) ** 1/2)
            
            fscore  = 1 
            for iscore in indv_score:
                fscore *= iscore if iscore else 1
            scores[fpath]   = fscore / len(contents)

    sorted_scores   = sorted(scores,key=lambda x: scores[x],reverse=True)

    for fpath in sorted_scores[:10]:
        print(f"{fpath}-> {scores[fpath]}")
    

def generate_news_articles(ds_root:str):
    if not os.path.exists(ds_root):
        os.mkdir(ds_root)

    root    = "C:/data/nlp/free-news-datasets-master/free-news-datasets-master/News_Datasets"
    for cat in [os.path.join(root,ccat) for ccat in os.listdir(root)]:
        print(f"parsing {cat}")
        for fname in [os.path.join(cat,name) for name in os.listdir(cat)]:

            #Load contents
            try:
                with open(fname,'r',encoding='utf_8') as readfile:
                    contents    = json.loads(readfile.read()) 
                    if contents['language'] == 'english':
                        text        = contents['title'] + "\n\n" + contents['text']

                        if text.count("https") > 30: #Skip too many links
                            continue
                        #Write to db 
                        with open(os.path.join(ds_root,str(random.randint(100_000_000,999_999_999))+".txt"),'w',encoding='utf_8') as writefile:
                            writefile.write(text)
            except PermissionError:
                pass


def generate_stack_overflow(ds_root:str):
    if not os.path.exists(ds_root):
        os.mkdir(ds_root)

    root    = "C:/data/nlp/stackoverflow"
    for fname  in [os.path.join(root,ccat) for ccat in os.listdir(root)]:
        with open(fname,'r',encoding='utf_8') as readfile:
            contents    = readfile.read()
            paragraphs  = contents.split("<p>")[1:]
            paragraphs  = [p.split("</p>")[0] for p in paragraphs]

            #Write to db 
            with open(os.path.join(ds_root,str(random.randint(100_000_000,999_999_999))+".txt"),'w',encoding='utf_8') as writefile:
                writefile.write("\n".join(paragraphs))


if __name__ == "__main__":
    vocab_name              = '32k_2'
    #generate_stack_overflow("C:/data/nlp/stackclean")
    #exit()
    #exit()
    #generate_news_articles("C:/data/nlp/newsarticles")
    #print(f"tokenization of '<|endoftext|>' -> {tokenizer.encode('<|endoftext|>').ids}")
    #prep_data_for_training(desired_sources=["C:/data/nlp/crawl"],final_dir="C:/data/nlp/training")#,"C:/data/nlp/gutenberg/books","C:/data/nlp/academic","C:/data/nlp/code/randpython_files"])
    train_tokenizer(vocab_size=32768,train_root="C:/data/nlp/training",name=vocab_name)
    tokenizer               = load_tokenizer(f"C:/data/nlp/{vocab_name}")
    create_token_file_parallel("C:/data/nlp/training",tokenizer,f"C:/data/nlp/tokens{vocab_name}/")
    #create_token_file("C:/data/nlp/training",tokenizer)
    exit()
