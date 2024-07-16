import torch 
from run import GPTSteinshark
from tokenizer import SteinTokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
import random

__DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#SETTINGS 
input_size  = 64
vocab_size  = 8192
embed_size  = 256
n_layers    = 2
n_heads     = 8  
temperature = .9

model       = GPTSteinshark(input_size=input_size,vocab_size=vocab_size,n_embed=embed_size,n_layer=n_layers,n_head=n_heads)
model.name                  = "steinshark1.model"
model.load_state_dict(torch.load(f"models/{model.name}"))
model.model.eval()


#Load Tokenizer
tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename="stein_tokenizer_bpe/vocab.json",merges_filename="stein_tokenizer_bpe/merges.txt")



with torch.no_grad():
    while True:

        start_text              = input("prompt: ")
        tokens                  = tokenizer.encode(start_text.lower()).ids
        print_buffer            = tokens 

        while not "<|endoftext|>" in start_text and not len(start_text) > 1024:
            #Encode start text
            context             = torch.tensor(tokens[:input_size],dtype=torch.long)

            #Send through model for logits
            output_logits       = model.forward(input_ids=context).logits[-1]

            #Softmax first 
            output_distr        = torch.nn.functional.softmax(output_logits,dim=0)
            temperature_distr   = torch.nn.functional.softmax(torch.pow(output_distr,temperature),dim=0)

            #Sample 
            next_token          = random.choices(list(range(tokenizer._tokenizer.get_vocab_size())),k=1,weights=temperature_distr)
            tokens              += next_token 
            print_buffer        += next_token

            if random.random() < 1/4:
                print(f"{tokenizer.decode(print_buffer)}",end='')
                print_buffer    = []


        #Print out final text

            
