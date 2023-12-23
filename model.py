from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, GPT2Tokenizer, GenerationConfig, Trainer
import torch    
from torch.utils.data import DataLoader, Dataset
import json 
import os 
import random 

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NotEnoughTextError(Exception):

    def __init__(self,msg:str):
        self.msg    = msg 
    
    def __repr__(self):
        return self.msg 
    

class GPTSteinsharkDataSet(Dataset):

    def __init__(self,ds_root:str):

        self.filenames  = [os.path.join(ds_root,file) for file in os.listdir(ds_root)]

        self.texts      = [open(fname,'r',encoding='utf_8').read() for fname in self.filenames]

    #returns text_ids and attention_mask
    def __getitem(self,i):
        #Pick random text 
        text    = random.choice(self.text)

class GPTSteinsharkTokenizer():

    def __init__(self,vocabulary: str|list|dict):
        self.base_chars     = set()
        if isinstance(vocabulary,str):
            with open(vocabulary,"r",encoding='utf_8') as file:
                self.words     = {word:i for i,word in enumerate(json.loads(file.read()))}
                self.tokens    = {value:key for key,value in self.words.items()}

        elif isinstance(vocabulary,list):
            self.tokens     = {i:item for i,item in enumerate(vocabulary)}
        elif isinstance(vocabulary,dict):
            self.tokens     = vocabulary
        
    
    def munch(self,candidate:str,text:str):
        candidate   += text[0]
        text        = text[1:]

        return candidate, text 
        

    def encode(self,text:str):
        encoding    = [] 

        while text:

            #iterate 
            candidate_token,text    = self.munch('',text)
            while candidate_token in self.words and text:
                candidate_token,text    = self.munch(candidate_token,text)
            
            if not candidate_token in self.words:
                candidate_token,text    = candidate_token[:-1],candidate_token[-1]+text
            
            if candidate_token:
                encoding.append(self.words[candidate_token])
        return encoding

    def encode_n(self,text:str,n:int):
        encoding    = [] 
        text_len    = len(text)

        while len(encoding) < n:

            #iterate 
            candidate_token,text    = self.munch('',text)

            #Munch until no longer in
            while candidate_token in self.words and text:
                candidate_token,text    = self.munch(candidate_token,text)
            
            #Check if no text:
            if not text:
                raise NotEnoughTextError(f"text len {text_len}")
            if not candidate_token in self.words:
                candidate_token,text    = candidate_token[:-1],candidate_token[-1]+text
            
            if candidate_token:
                encoding.append(self.words[candidate_token])

        return encoding

    def decode(self,tokens:list[int]):
        return ''.join([self.tokens[token] for token in tokens])



class GPTSteinshark(GPT2LMHeadModel):

    def __init__(self,
                 input_size=1024,
                 vocab_size=32768,
                 n_embed=768,
                 n_layer=12,
                 n_head=12,
                 act_fn="gelu_new"

                 ):
        
        #Create config for the model 
        self.config             = GPT2Config(vocab_size=vocab_size,
                                             n_positions=input_size,
                                             n_embd=n_embed,
                                             n_layer=n_layer,
                                             n_head=n_head,
                                             activation_function=act_fn
                                             )

        #Create the model
        super(GPTSteinshark,self).__init__(self.config)
        self.model              = GPT2LMHeadModel(self.config).to(DEVICE)

    def train(self,dataloader:DataLoader):



if __name__ == "__main__":

    #tokenizer   = GPTSteinsharkTokenizer('vocabulary.txt')
    #tokenizer.words['x']   = len(tokenizer.words)+1
    #tokenizer.tokens[len(tokenizer.words)]  = 'x'
    ##tokenizer.tokens['x']   = len(tokenizer.tokens)+1
    #print(f"{tokenizer.encode('this is a sample text')}\t-> '{tokenizer.decode(tokenizer.encode('this is a sample text'))}'")

    model   = GPTSteinshark()
    out     = model.transformer.forward(input_ids=torch.tensor([[122,65,512]]),attention_mask=torch.tensor([[1,1,1]]),return_dict=False,output_hidden_states=False)
    m       = torch.nn.Linear(768,1024)
    print(f"logits are {out[0].shape}, vs: {m(out[0]).shape}")
    #tok     = GPT2Tokenizer.from_pretrained('gpt2')
    #print(model._get_generation_mode(GenerationConfig.from_model_config(model.config),None))
    #Model generate expects an 'input_ids' and 'attention_mask' arg. 'max_new_tokens' also 
