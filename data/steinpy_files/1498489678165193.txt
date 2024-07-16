import torch 
from run import GPTSteinshark
from tokenizer import SteinTokenizer


__DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#SETTINGS 
input_size  = 64
bs          = 4
cutoff      = None
epochs      = 8
lr          = 2e-4
save_ckpt   = 256 
vocab_size  = 512
n_embed     = 512
model       = GPTSteinshark()
tokenizer   = GPTSteinshark(input_size=input_size,vocab_size=vocab_size,n_embed=n_embed)


#BUILD TOKENIZER 


model_in    = tokenizer("I have a cat. My cat is:",return_tensors="pt")
model_in    = {key:model_in[key].to(__DEVICE) for key in model_in}
outputs     = model.generate(**model_in,max_length=input_size)


text        = tokenizer.decode(outputs[0])
print(f"output: {text}")
