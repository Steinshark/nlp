import torch 
from torch.utils.data import Dataset
import os 
from transformers import PreTrainedTokenizer
import random

def load_data(ds_root:str,chunk_size:int,tokenizer:PreTrainedTokenizer,cutoff=None,eval_split=.1,replace_newline=True):

    fulltext = ''
    
    #Get text
    for item in os.listdir(ds_root): 
        item        = os.path.join(ds_root,item)
        full_text   = open(item,"r",encoding='utf-8').read()
        if replace_newline:
            full_text = full_text.replace("\n"," ")
        fulltext += full_text
        # while len(full_text) > chunk_size:
        #     data.append(full_text[:chunk_size])
        #     full_text   = full_text[chunk_size:]

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