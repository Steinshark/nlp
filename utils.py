import torch 
from torch.utils.data import Dataset
import os 
from transformers import PreTrainedTokenizer
import random
import re
import nltk
from nltk import corpus

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

    with open("C:/code/nlp/data/brown.txt","w",encoding='utf-8') as file:
        file.write(corpus.abc.raw().replace("\n\n","<|endoftext|>"))

if __name__ == "__main__":
    get_readmes()