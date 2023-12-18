from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
import torch


__DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPTSteinsharkTokenizer():

    def __init__(self,vocab_size):
        self.base_chars     = set() 
        self.tokens         = list(range(vocab_size))
        
    def generate_vocab_from_corpus(filelist):

        text_raw    = ''
        #Read from each file
        for file in filelist:

            #Get raw text 
            with open(file,"r",encoding="utf-8") as file:
                rawtext     = file.read()
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
