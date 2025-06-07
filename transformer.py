import torch 
from collections import OrderedDict
import math
import random 
import os 
import json
import numpy 
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

os.environ['TORCH_USE_CUDA_DSA'] = "True"

def apply_rope(x, seq_len, device):

    head_dim = x.size(-1)
    half_dim = head_dim // 2

    # Compute frequencies
    theta = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.bfloat16, device=device) / half_dim))
    seq_idx = torch.arange(seq_len, device=device)#.float()
    freqs = torch.einsum("i,j->ij", seq_idx, theta)

    sin = freqs.sin()[None, None, :, :]
    cos = freqs.cos()[None, None, :, :]

    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return x_rotated

class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self, embed_dim, num_heads,n_positions, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(MultiHeadAttention, self).__init__()
        
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert embed_dim % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.n_positions        = n_positions
        self.embed_dim          = embed_dim # Model's dimension
        self.num_heads          = num_heads # Number of attention heads
        self.d_k                = embed_dim // num_heads # Dimension of each head's key, query, and value
        self.device             = device
        # Linear layers for transforming inputs
        self.layer_1            = torch.nn.Linear(embed_dim,embed_dim*3,bias=True)
        self.W_o                = torch.nn.Linear(embed_dim, embed_dim,device=device,bias=True)     # Output transformation
        self.scale              = 1 / math.sqrt(self.d_k)
        self.register_buffer('mask',torch.tril(torch.ones(n_positions,n_positions,device=device)))
        

  
    def forward(self, x:torch.Tensor):
        B, N, C         = x.size()

        #Create mask size 
        #mask            = torch.tril(torch.ones(self.n_positions,self.n_positions,device=self.device,requires_grad=False))

        # Apply linear transformations and split heads
        Q,K,V           = self.layer_1(x).split(self.embed_dim,dim=2)
        Q:torch.Tensor  = Q.view(B, N, self.num_heads, self.d_k).transpose(1,2)
        K:torch.Tensor  = K.view(B, N, self.num_heads, self.d_k).transpose(1,2)
        V:torch.Tensor  = V.view(B, N, self.num_heads, self.d_k).transpose(1,2)

        # Apply RoPE
        Q               = apply_rope(Q,N,self.device)
        K               = apply_rope(K,N,self.device)

        
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,SDPBackend.FLASH_ATTENTION,SDPBackend.MATH,SDPBackend.CUDNN_ATTENTION]):
            attn_out    = scaled_dot_product_attention(Q,K,V,dropout_p=.05,is_causal=True,scale=self.scale)
            attn_out    = attn_out.transpose(1,2).contiguous().view(B,N,C)
            output      = self.W_o(attn_out)
            return output
            
        # #     return output
        # # Perform scaled dot-product attention
        # attn_scores     = Q @ K.transpose(-2,-1)
        # attn_scores     = attn_scores * (1 / math.sqrt(self.d_k))
        # attn_scores.masked_fill_(self.mask[:N,:N]==0,float("-inf"))
        # attn_scores     = torch.nn.functional.softmax(attn_scores,dim=-1)

        # attn_result     = attn_scores @ V 
        # attn_result     = attn_result.transpose(1,2).contiguous().view(B,N,C)
        # output          = self.W_o(attn_result)
        # return output


class DecoderLayer(torch.nn.Module):

    def __init__(self,n_embed,n_head,n_ff,dropout=.1,act_fn=torch.nn.GELU,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),n_positions=512):
        super(DecoderLayer,self).__init__()
        
        #Self attention
        self.mh_attn                = MultiHeadAttention(n_embed,n_head,n_positions,device=device)
        self.mha_dropout            = torch.nn.Dropout(p=dropout)
        self.mha_layer_norm         = torch.nn.LayerNorm(n_embed,device=device,)
        
        
        #Linear 
        self.ff_layers              = torch.nn.Sequential(
            torch.nn.Linear(n_embed,n_ff,device=device),
            act_fn(),
            torch.nn.Linear(n_ff,n_embed,device=device))
        self.ff_dropout             = torch.nn.Dropout(p=dropout)
        self.ff_layer_norm          = torch.nn.LayerNorm(n_embed,device=device)

   
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



class EncoderBlock(torch.nn.Module):
    
    #Context_dim should be n_embed//4
    def __init__(self,n_vocab,embed_dim,n_positions,device):
        super(EncoderBlock,self).__init__()

        self.device                 = device
        #Start with a separate context embeddings module 
        self.semantic_embeddings    = torch.nn.Embedding(n_vocab,embed_dim,device=device)
        #self.input_pos_embeddings   = torch.nn.Embedding(n_positions,embed_dim,device=device)          


    #Given the context ids and the input ids, return the embeddings to be passed forward 
    def forward(self,input_ids:torch.Tensor)->torch.Tensor:
        
        #Compute actual input embeddings
        semantic_embeddings:torch.Tensor        = self.semantic_embeddings(input_ids)

        return semantic_embeddings
       
            


class LMSteinshark(torch.nn.Module):


    def __init__(self,
                 n_positions:int=512,
                 n_embed    :int=512,
                 n_layers   :int=16,
                 n_heads    :int=16,
                 n_ff       :int=1024,
                 n_vocab    :int=32768,
                 act_fn     :torch.nn.functional=torch.nn.GELU,
                 dropout    :float=.1):
        

        super(LMSteinshark,self).__init__()
        
        #Make checks 
        assert n_embed % n_heads == 0

        
        #Set class variables
        self.n_positions            = n_positions
        self.n_embed                = n_embed
        self.n_layers               = n_layers
        self.n_heads                = n_heads
        self.n_ff                   = n_ff
        self.devices                = [torch.device('cuda:0'),torch.device('cuda:0')]
        self.device                 = self.devices[0]

        #Use learnable position embeddings
        self.input_block            = EncoderBlock(n_vocab,n_embed,n_positions,self.devices[0])

        #Create decoder stacks 
        self.transformer_stack      = torch.nn.Sequential(
            OrderedDict(
                {str(i):DecoderLayer(n_embed,n_heads,n_ff,dropout=dropout,act_fn=act_fn,device=self.devices[0],n_positions=n_positions) 
                 for i in range(n_layers)})).to(self.devices[0])
        
        #Layer norm one last time on the LM Heada
        self.output_ln              = torch.nn.LayerNorm(n_embed).cuda()

        #Calc params 
        self.n_params               = sum(p.numel() for p in self.parameters())

        #Save name 
        self.name                   = "LMStein"

        #Stats 
        self.stats                  = {"iter_through":0,
                                       "tok_through":0,
                                       "eps_through":0,
                                       "losses":[],
                                       "tok_snap":[]}
        #Init weights 
        self.initialize_weights()
        

        #prep template tensor for oversize inputs 

    
    def forward(self,input_ids:torch.Tensor,target_ids:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        
        #Split tokens
        #input_seq,targets               = self.split_input(input_ids,target_ids)

        x                               = self.input_block(input_ids) #returns (bs,T,n_embed)

        #Pass through transformer stack
        x                               = self.transformer_stack(x)

        x                               = self.output_ln(x)
        logits                          = x @ self.input_block.semantic_embeddings.weight.T
        #Pass through lm_head to get logits
        return logits, target_ids

   
    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.kaiming_normal_(param)


    def create_embeddings(self,input_ids:torch.Tensor,target_ids:torch.Tensor):
        #Get semantic embeddings regardless 
        semantic_embeddings         = self.semantic_embedder(input_ids)


        #Check to combine semantics if need be
        while semantic_embeddings.size(1) > self.n_positions:
            
            #If can be combined from a 
            #Combine a semantic_embedding vector with another and reduce the matrix
            r_index                             = random.randint(0,semantic_embeddings.size(1)-2) #1 for index, 1 to recombine after 
            combined_sum                        = torch.sum(semantic_embeddings[:,r_index:r_index+2,:],dim=1)
            target_ids                          = torch.cat([target_ids[:,:r_index+1],target_ids[:,r_index+2:]],dim=1)

            semantic_embeddings[:,r_index,:]    = combined_sum
            semantic_embeddings                 = torch.cat([semantic_embeddings[:,:r_index+1,:],semantic_embeddings[:,r_index+2:,:]],dim=1)


        #Get position embeddings 
        position_idx                = torch.arange(min(input_ids.size(1),self.n_positions),device=self.devices[0]).unsqueeze(0).expand_as(input_ids if input_ids.size(1) <= self.n_positions else torch.zeros(input_ids.size(0),self.n_positions))
        position_embeddings         = self.position_embedder(position_idx)
        
        
        #Return the sum
        batch_embeds                = position_embeddings + semantic_embeddings
        return batch_embeds, target_ids
        
        #Group tokens into max of n_positions 
        raise NotImplementedError("Over-Sized context not implemented. Get to work!")
        
    
    def create_pos_embeddings(self,period=5_000):
        idx_matrix          = [[0 for embd_idx in range(self.n_embed)] for tok_idx in range(self.n_positions)]
        
        for tok_idx,tok_embd in enumerate(idx_matrix):
            for embd_idx,_ in enumerate(tok_embd):

                #Populate 
                if embd_idx % 2 == 0:
                    val     = math.sin((tok_idx+1)/(pow(period,(embd_idx//2)/self.n_embed)))
                else:
                    val     = math.cos((tok_idx+1)/(pow(period,(embd_idx//2)/self.n_embed)))
                
                idx_matrix[tok_idx][embd_idx]   = val

        embeddings          = torch.tensor(idx_matrix,device=self.devices[0])
        
        return embeddings


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


    def generate(self,prompt:list[int],tokenizer:ByteLevelBPETokenizer,n_tokens=128,temperature=.5,top_k=30):
        self.eval()

        with torch.no_grad():
            tokens          = prompt
            model_output    = []
        
            while len(tokens) - len(prompt) < n_tokens:

                input_seq                       = torch.tensor(tokens[-self.n_positions:],device=self.device).long().unsqueeze_(0)
                target_ids                      = torch.empty_like(input_seq)
                logits                          = self(input_seq,target_ids)[0]

                #Scale logits by temp 
                logits                          = logits / temperature
                top_val,top_i                   = torch.topk(logits,k=top_k)

                distribution                    = torch.nn.functional.softmax(top_val,dim=-1)
                next_tokens_i                   = torch.distributions.Categorical(probs=distribution).sample()
                next_tokens                     = torch.gather(top_i,2,next_tokens_i.unsqueeze(-1)).squeeze(-1)
                
                next_token                      = next_tokens[-1,-1]
                model_output.append(next_token)

                #Stop with end seq
                if next_token == tokenizer.encode("<|endoftext|>").ids[0]:
                    self.train()
                    return model_output
                tokens                          = tokens + [next_token]

        
        self.train()
        return model_output


    def split_input(self,input_ids:torch.Tensor,target_ids:torch.Tensor=None):
        input_seq                   = input_ids[...,-self.n_positions:].contiguous()
        target_ids                  = target_ids[...,-self.n_positions:].contiguous()
        
        return input_seq,target_ids


    def model_info(self) -> str:
        info    = f"Model: {self.name}\n"

        info    += f'parameters:\t{self.n_params // 1_000_000}M\n'
        info    += f'num layers:\t{self.n_layers}\n'
        info    += f'context:\t{self.n_positions}\n'
        info    += f'embed dim:\t{self.n_embed}\n'
        info    += f"ff size:\t{self.n_ff}\n"
        info    += f'num heads:\t{self.n_heads}\n'

        return info

if __name__ == "__main__":

    n_embed     = 1536
    n_ff        = n_embed*2
    n_heads     = n_embed//128
    bs          = 1
    n_positions = 256
    n_vocab     = 32768
    n_layers    = 4

    #Create model
    lm          = LMSteinshark(n_layers=n_layers,n_embed=n_embed,n_heads=n_heads,n_positions=n_positions,n_vocab=n_vocab,n_ff=n_ff,dropout=0)
    lm.name     = '4x1536x256x2'
    import training
    lm.load(training.MODELS)
    print(lm.model_info())
    import tok 
    t           = tok.load_tokenizer('32k') 

    while True:
        prompt      = input('user: ')
        tokens      = t.encode(prompt).ids
        output      = t.decode(lm.generate(tokens,t,64,.7,top_k=100))
        print(f"\n\nmodel:{output}\n\n\n")
    exit()
