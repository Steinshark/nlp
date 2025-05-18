import torch 
from transformer import ContextEncoderBlock

dev     = torch.device('cuda')
ceb     = ContextEncoderBlock(32768,32,1024,dev)
ctxt    = torch.randint(0,32768-1,(8,67)).to(dev)
inp     = torch.randint(0,32768-1,(8,17)).to(dev)
ceb(ctxt,inp)