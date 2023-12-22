from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
import torch
import json 

__DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPTSteinsharkTokenizer():

    def __init__(self,vocabulary: str|list|dict):
        self.base_chars     = set()
        if isinstance(vocabulary,str):
            with open(vocabulary,"r",encoding='utf_8') as file:
                self.tokens     = json.loads(file.read())
        elif isinstance(vocabulary,list):
            self.tokens     = {i:item for i,item in enumerate(vocabulary)}
        elif isinstance(vocabulary,dict):
            self.tokens     = vocabulary
        
        
    def generate_vocab_from_corpus(filelist):

        text_raw    = ''
        #Read from each file
        for file in filelist:

            #Get raw text 
            with open(file,"r",encoding="utf-8") as file:
                rawtext += file.read()
                pairings    = {}
                

                #Get all positions
                for i in range(len(rawtext)-1):
                    pair    =f"{pairings[rawtext[i]]}{pairings[rawtext[i+1]]}" 
                    if pair in pairings:
                        pairings[pair] += 1
                    else:
                        pairings[pair]  = 1 
                        
    

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
        super(GPTSteinshark,self).__init__(self.config).to(__DEVICE)
        self.model              = GPT2LMHeadModel(self.config)
