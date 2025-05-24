import torch 
import torch.nn as nn
from collections import OrderedDict
import math
import random 
import os 
import json
import numpy 
from tokenizers.implementations import ByteLevelBPETokenizer

os.environ['TORCH_USE_CUDA_DSA'] = "True"


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, tfmr_input_size, device=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tfmr_input_size = tfmr_input_size

        self.layer_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("mask", torch.tril(torch.ones(tfmr_input_size, tfmr_input_size)))
        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param)

    def split_heads(self, x):
        B, T, D = x.size()
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]

    def combine_heads(self, x):
        B, H, T, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        #attn = torch.softmax(scores, dim=-1)
        attn    = torch.nn.functional.softmax(scores,dim=-1)
        return torch.matmul(attn, V)

    def forward(self, x):
        B, T, _ = x.size()
        qkv = self.layer_qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Create causal mask
        mask = self.mask[:T, :T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self.combine_heads(attn_output)
        return self.W_o(attn_output)



class DecoderLayer(torch.nn.Module):

    def __init__(self,n_embed,n_head,n_ff,dropout=.1,act_fn=torch.nn.GELU,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),tfmr_input_size=512):
        super(DecoderLayer,self).__init__()
        
        #Self attention
        self.mh_attn                = MultiHeadAttention(n_embed,n_head,tfmr_input_size,device=device)
        self.mha_dropout            = torch.nn.Dropout(p=dropout)
        self.mha_layer_norm         = torch.nn.LayerNorm(n_embed,device=device)
        
        
        #Linear 
        self.ff_layers              = torch.nn.Sequential(
            torch.nn.Linear(n_embed,n_ff,device=device),
            act_fn(),
            torch.nn.Linear(n_ff,n_embed,device=device))
        self.ff_dropout             = torch.nn.Dropout(p=dropout)
        self.ff_layer_norm          = torch.nn.LayerNorm(n_embed,device=device)

        self.initialize_weights()
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        
        #Apply MHA, residual connection, and layer_norm
        attn_output                 = self.mh_attn(self.mha_layer_norm(x))
        attn_output                 = self.mha_dropout(attn_output)
        x                           = x + attn_output

        #Apply ff_layer, residual, and layer_norm
        ff_norm                     = self.ff_layer_norm(x)
        ff_output                   = self.ff_layers(ff_norm)
        ff_output                   = self.ff_dropout(ff_output)
        x                           = x + ff_output

        return x

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)



class Summarizer(torch.nn.Module):

    def __init__(self,context_size,embed_dim,num_heads,n_tok,n_vocab,device,p=.1):
        super(Summarizer,self).__init__()
        self.embed_dim                      = embed_dim
        self.device                         = device 
        self.embeddings_sem                 = torch.nn.Embedding(n_vocab,embed_dim)
        self.embeddings_pos                 = torch.nn.Embedding(context_size,embed_dim)

        self.attn                           = torch.nn.MultiheadAttention(embed_dim,num_heads,p,bias=True,batch_first=True)
        self.summary_tokens                 = torch.nn.Parameter(torch.randn(1,n_tok,embed_dim))
        self.l_norm                         = torch.nn.LayerNorm(embed_dim)
        self.n_vocab                        = n_vocab
        self.to(device)


    def embed(self,input_seq:torch.Tensor):
        input_seq                           = input_seq.long()
        sem_embd                            = self.embeddings_sem(input_seq)
        
        B,T                                 = input_seq.shape
        input_position_idx                  = torch.arange(T, device=self.device).unsqueeze(0).expand(B,T)
        pos_emb                             = self.embeddings_pos(input_position_idx)

        embedded_seq                        = sem_embd + pos_emb

        #(BS,N,E)
        return embedded_seq

    def forward(self, input_seq:torch.Tensor):

        #Embed the input_sequence 
        #input(f"vocab is {self.n_vocab} -> in max is {input_seq.max()}")
        B,T                                 = input_seq.shape
        inputs_embedded                     = self.embed(input_seq)

        attn_input                          = self.summary_tokens.expand(B,-1,-1)
        attn_out                            = self.attn(attn_input,inputs_embedded,inputs_embedded)[0]


        return self.l_norm(attn_out)



class TinySummarizer(torch.nn.Module):
    def __init__(self, embed_size, max_context, n_heads, summary_tokens):
        super().__init__()

        #Embed differently for the summary tokens
        self.summary_embeddings             = torch.nn.Embedding(n_vocab,embed_size) 
        self.summary_pos_embeddings         = torch.nn.Embedding(max_context,embed_size)
        self.attn = torch.nn.MultiheadAttention(embed_size, n_heads, batch_first=True)
        self.summary_tokens = torch.nn.Parameter(torch.randn(1, summary_tokens, embed_size))

    def forward(self, x):
        B = x.size(0)
        summary_input = self.summary_tokens.expand(B, -1, -1)
        summary_out, _ = self.attn(summary_input, x, x)
        return summary_out  # [B, summary_tokens, D]
   


class MiniTransformerSteinshark(torch.nn.Module):


    def __init__(self,
                 context1_size  :int=512,
                 context2_size  :int=256,
                 core_size      :int=256,
                 lg_size        :int=64,
                 md_size        :int=64,
                 n_embed        :int=512,
                 n_layers       :int=16,
                 n_heads        :int=16,
                 n_ff           :int=1024,
                 n_vocab        :int=32768,
                 act_fn         :torch.nn.functional=torch.nn.GELU,
                 dropout        :float=.1):
        

        super(MiniTransformerSteinshark,self).__init__()
        print(f"layers is {n_layers}")
        #Make checks 
        assert n_embed % n_heads == 0

        
        #Set class variables
        self.context1_size          = context1_size
        self.context2_size          = context2_size
        self.core_size              = core_size
        self.lg_size                = lg_size
        self.md_size                = md_size
        self.tfmr_input_size        = self.core_size + self.lg_size + self.md_size

        self.n_embed                = n_embed
        self.n_layers               = n_layers
        self.n_heads                = n_heads
        self.n_ff                   = n_ff
        self.device                 = torch.device('cuda:0')
        self.dropout                = dropout

        #Each block returns a sequence of embedded inputs
        self.summarizer_lg:Summarizer           = Summarizer(self.context1_size,self.n_embed,num_heads=self.n_heads,n_tok=self.lg_size,n_vocab=n_vocab,device=self.device)
        self.summarizer_md:Summarizer           = Summarizer(self.context2_size,self.n_embed,num_heads=self.n_heads,n_tok=self.md_size,n_vocab=n_vocab,device=self.device)
        self.embeddings                         = torch.nn.Embedding(n_vocab,n_embed).to(self.device)
        self.embeddings_pos                     = torch.nn.Embedding(self.core_size,n_embed).to(self.device)

        #the follow-on transformer stacks 
        self.transformer_stack                  = torch.nn.Sequential(OrderedDict({str(i):DecoderLayer(self.n_embed,self.n_heads,self.n_ff,dropout=self.dropout,act_fn=act_fn) for i in range(self.n_layers)})).to(self.device)        

        #Allow for language modelling 
        self.lm_head                            = torch.nn.Sequential(torch.nn.LayerNorm(n_embed),torch.nn.Linear(n_embed,n_vocab,bias=True,device=self.device)).to(self.device)

        #Calc params 
        self.n_params                           = sum(p.numel() for p in self.parameters())

        #Save name 
        self.name                               = "Attempt1"

        #Stats 
        self.stats                              = { "iter_through":0,
                                                    "tok_through":0,
                                                    "eps_through":0,
                                                    "losses":[],
                                                    "tok_snap":[]}
        #Init weights 
        self.initialize_weights()
        

        #prep template tensor for oversize inputs 


    def forward(self,input_ids:torch.Tensor,target_ids:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        
        #Ensure we have input as Long dtype
        input_ids                           = input_ids.long()

        #Create 3 segments from the input ids 
        lg_context_ids                      = torch.clone(input_ids[:,-(self.context1_size+self.context2_size+self.core_size):-(self.context2_size+self.core_size)]).contiguous()   #Use everything up to actual sequence
        md_context_ids                      = torch.clone(input_ids[:,-(self.context2_size+self.core_size):-self.core_size]).contiguous()
        input_seq                           = torch.clone(input_ids[:,-self.core_size:]).contiguous()
        target_ids                          = target_ids[:,-self.core_size:].contiguous()

        #Create summary sequences
        lg_context_embeddings:torch.Tensor  = self.summarizer_lg(lg_context_ids)        #n_tok_lg
        md_context_embeddings:torch.Tensor  = self.summarizer_md(md_context_ids)        #n_tok_md 
        
        #Create core sequence
        B,T                                 = input_seq.shape
        sem_emb:torch.Tensor                = self.embeddings(input_seq)
        input_position_idx                  = torch.from_numpy(numpy.asarray( [numpy.arange(T) for _ in range(B)])).to(self.device).long()
        pos_emb                             = self.embeddings_pos(input_position_idx)
        input_embeddings:torch.Tensor       = sem_emb + pos_emb

        transformer_input                   = torch.cat((lg_context_embeddings,md_context_embeddings,input_embeddings),1)#.to(self.transformer_dev)   #(B,T,E)
        transformer_output                  = self.transformer_stack(transformer_input)

        logits:torch.Tensor                 = self.lm_head(transformer_output)[:,-self.core_size:,:]

        #Pass through lm_head to get logits
        return logits.contiguous(), target_ids.contiguous()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.kaiming_normal_(param)


    def save(self,root="C:\\data\\nlp\\models"):
        save_path       = os.path.join(root,self.name)
 
        #save metadata
        with open(save_path+f".json",'w') as writefile:
            writefile.write(json.dumps(self.stats))
        
        #Save params
        torch.save(self.state_dict(),save_path+f".pt")
        print(f"\n\nSaved model\n\n")


    def load(self,root="C:\\data\\nlp\\models"):
        save_path       = os.path.join(root,self.name)

        #Load metadata
        with open(save_path+f".json",'r') as readfile:
            self.stats  = json.loads(readfile.read())
        
        #Load params
        self.load_state_dict(torch.load(save_path+f".pt",weights_only=True))
        print(f"\n\nLoaded model\n\n")


    def generate(self,prompt:list[int],tokenizer:ByteLevelBPETokenizer,n_tokens=128,temperature=.5):
        self.eval()

        with torch.no_grad():
            tokens  = prompt
        
            while len(tokens) - len(prompt) < n_tokens:

                input_seq                       = torch.tensor(tokens,device=self.device).long().unsqueeze_(0)
                context_seq                     = torch.empty_like(input_seq)
                logits                          = self(input_seq,context_seq)[0][0,-1,:]
                distribution                    = torch.nn.functional.softmax(logits/temperature,dim=-1)
                next_token                      = torch.multinomial(distribution,1)

                #Stop with end seq
                if next_token == tokenizer.encode("<|endoftext|>"):
                    self.train()
                    return tokens
                tokens                          = tokens + [next_token]

        
        self.train()
        return tokens





if __name__ == "__main__":

    n_embed     = 4 
    n_ff        = n_embed*2 
    n_heads     = n_embed//2
    bs          = 8 
    tfmr_input_size = 4
    n_vocab     = 32768

    #Create model
    lm          = MiniTransformerSteinshark(n_embed=n_embed,n_heads=n_heads,tfmr_input_size=tfmr_input_size,n_vocab=n_vocab,n_ff=n_ff,dropout=.1)
    from dataset import TokenizedDataset
    import numpy 
    toks        = numpy.load("C:/data/nlp/tokens0.npy")
    aa          = TokenizedDataset(toks,tfmr_input_size)
    batch       = aa.sample(1,6,lm.devices)
    lm.forward(batch['input_ids'],batch['target_ids'])
