import torch 
from collections import OrderedDict
import math
import random 
import os 
import json

class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self, embed_dim, num_heads,n_positions, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(MultiHeadAttention, self).__init__()
        
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert embed_dim % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.embed_dim          = embed_dim # Model's dimension
        self.num_heads          = num_heads # Number of attention heads
        self.d_k                = embed_dim // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q                = torch.nn.Linear(embed_dim, embed_dim,device=device,bias=True)    # Query transformation
        self.W_k                = torch.nn.Linear(embed_dim, embed_dim,device=device,bias=True)    # Key transformation
        self.W_v                = torch.nn.Linear(embed_dim, embed_dim,device=device,bias=True)    # Value transformation
        self.W_o                = torch.nn.Linear(embed_dim, embed_dim,device=device,bias=True)     # Output transformation

        self.register_buffer('mask',torch.tril(torch.ones(n_positions,n_positions,device=device)))
        
        self.initialize_weights()


    def scaled_dot_product_attention(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):

        # Scores is a n_heads x  n_position x n_position matrix -> [B,n_head,n_pos,n_pos]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)        
        attn_scores = torch.masked_fill(attn_scores,self.mask[:attn_scores.size(-2),:attn_scores.size(-2)]==0,float("-inf"))
        attn_norm   = torch.softmax(attn_scores, dim=-1)

        #print(f"norm:\n{attn_norm[0,0,:4,:4]}")
        output = torch.matmul(attn_norm, V)
        #input(f"out is {output.shape}")
        
        return output
        
    def split_heads(self, x:torch.Tensor):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x:torch.Tensor):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
    def forward(self, x:torch.Tensor):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
    def initialize_weights(self):

        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.kaiming_normal_(param)


class MultiHeadAttention2(torch.nn.Module):
    
    def __init__(self, embed_dim, num_heads,n_positions, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(MultiHeadAttention2, self).__init__()
        
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert embed_dim % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.embed_dim          = embed_dim # Model's dimension
        self.num_heads          = num_heads # Number of attention heads
        self.d_k                = embed_dim // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.layer_1            = torch.nn.Linear(embed_dim,embed_dim*3,bias=True)
        self.W_o                = torch.nn.Linear(embed_dim, embed_dim,device=device,bias=True)     # Output transformation

        self.register_buffer('mask',torch.tril(torch.ones(n_positions,n_positions,device=device)))
        
        self.initialize_weights()


    def scaled_dot_product_attention(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):

        # Scores is a n_heads x  n_position x n_position matrix -> [B,n_head,n_pos,n_pos]    
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (1/ math.sqrt(self.d_k))
        attn_scores = torch.masked_fill(attn_scores,self.mask[:attn_scores.size(-2),:attn_scores.size(-2)]==0,float("-inf"))
        attn_norm   = torch.softmax(attn_scores, dim=-1)

        #print(f"norm:\n{attn_norm[0,0,:4,:4]}")
        output = torch.matmul(attn_norm, V)
        #input(f"out is {output.shape}")
        
        return output
        
    def split_heads(self, x:torch.Tensor):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x:torch.Tensor):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
    def forward(self, x:torch.Tensor):
        B, N, C         = x.size()

        # Apply linear transformations and split heads
        Q,K,V           = self.layer_1(x).split(self.embed_dim,dim=2)
        Q:torch.Tensor  = Q.view(B, N, self.num_heads, C // self.num_heads).transpose(1,2)
        K:torch.Tensor  = K.view(B, N, self.num_heads, C // self.num_heads).transpose(1,2)
        V:torch.Tensor  = V.view(B, N, self.num_heads, C // self.num_heads).transpose(1,2)
        
        # Perform scaled dot-product attention
        attn_scores     = Q @ K.transpose(-2,-1)
        attn_scores     = attn_scores * (1 / math.sqrt(self.embed_dim))
        attn_scores.masked_fill_(self.mask[:N,:N]==0,float("-inf"))
        attn_scores     = torch.nn.functional.softmax(attn_scores,dim=-1)
        attn_result     = attn_scores @ V 
        attn_result     = attn_result.transpose(1,2).contiguous().view(B,N,C)

        output          = self.W_o(attn_result)
        return output
    
    def initialize_weights(self):

        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.kaiming_normal_(param)


class DecoderLayer(torch.nn.Module):

    def __init__(self,n_embed,n_head,n_ff,dropout=.1,act_fn=torch.nn.GELU,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),n_positions=512):
        super(DecoderLayer,self).__init__()
        
        #Self attention
        self.mh_attn                = MultiHeadAttention2(n_embed,n_head,n_positions,device=device)
        self.mha_dropout            = torch.nn.Dropout(p=dropout,inplace=True)
        self.mha_layer_norm         = torch.nn.LayerNorm(n_embed,device=device)
        
        
        #Linear 
        self.ff_layers              = torch.nn.Sequential(
            torch.nn.Linear(n_embed,n_ff,device=device),
            act_fn(),
            torch.nn.Linear(n_ff,n_embed,device=device))
        self.ff_dropout             = torch.nn.Dropout(p=dropout,inplace=True)
        self.ff_layer_norm          = torch.nn.LayerNorm(n_embed,device=device)

        self.initialize_weights()
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        
        #Apply MHA, residual connection, and layer_norm
        attn_output                 = self.mh_attn(self.mha_layer_norm(x))
        attn_output                 = self.mha_dropout(attn_output)
        x                           = x + attn_output

        #Apply ff_layer, residual, and layer_norm
        ff_output                   = self.ff_layers(self.ff_layer_norm(x))
        ff_output                   = self.ff_dropout(ff_output)
        x                           = x + ff_output

        return x

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)


class LMSteinshark(torch.nn.Module):


    def __init__(self,
                 n_positions:int=512,
                 n_embed    :int=512,
                 n_layers   :int=16,
                 n_heads    :int=16,
                 n_ff       :int=1024,
                 n_vocab    :int=32768,
                 act_fn     :torch.nn.functional=torch.nn.GELU,
                 dropout    :float=.1,
                 trig_embd  :bool=True):
        

        super(LMSteinshark,self).__init__()
        
        #Make checks 
        assert n_embed % n_heads == 0

        
        #Set class variables
        self.n_positions            = n_positions
        self.n_embed                = n_embed
        self.n_layers               = n_layers
        self.n_heads                = n_layers
        self.n_ff                   = n_ff
        self.devices                = [torch.device('cuda:0'),torch.device('cuda:1')]

        #Use learnable position embeddings
        self.semantic_embedder      = torch.nn.Embedding(n_vocab,n_embed,device=self.devices[0])
        if trig_embd:
            self.position_embedder  = lambda x : self.create_pos_embeddings(period=10_000)
        else:
            self.position_embedder  = torch.nn.Embedding(n_positions,n_embed,device=self.devices[0])

        #Create decoder stacks 
        self.device_0_modules       = torch.nn.Sequential(OrderedDict({str(i):DecoderLayer(n_embed,n_heads,n_ff,dropout=dropout,act_fn=act_fn,device=self.devices[0],n_positions=n_positions) for i in range(n_layers//2+2)})).to(self.devices[0])
        self.device_1_modules       = torch.nn.Sequential(OrderedDict({str(i):DecoderLayer(n_embed,n_heads,n_ff,dropout=dropout,act_fn=act_fn,device=self.devices[1],n_positions=n_positions) for i in range(n_layers//2-2)})).to(self.devices[1])
        
        #self.decoder_stack          = torch.nn.ModuleList([DecoderLayer(n_embed,n_heads,n_ff,dropout=dropout,act_fn=act_fn,device=self.devices[0],n_positions=n_positions//2) for _ in range(n_layers)] + [torch.nn.LayerNorm(n_embed,device=self.devices[0])])
        
        self.lm_head                = torch.nn.Sequential(torch.nn.LayerNorm(n_embed),torch.nn.Linear(n_embed,n_vocab,bias=False,device=self.devices[1])).to(self.devices[1])

        #Calc params 
        self.n_params               = sum(p.numel() for p in self.parameters())

        #Save name 
        self.name                   = "LMStein"

        #Stats 
        self.stats                  = {"iter_through":0,
                                       "tok_through":0,
                                       "eps_through":0,
                                       "losses":[]}
        #Init weights 
        self.initialize_weights()
        

        #prep template tensor for oversize inputs 


    def forward(self,input_ids:torch.Tensor,target_ids:torch.Tensor):

        #Convert tokens to embedding space 
        x,y                         = self.create_embeddings(input_ids,target_ids)
        #Pass through transformer stack
        x                           = self.device_0_modules(x)
        x                           = x.to(self.devices[1])
        x                            = self.device_1_modules(x)
        #x = self.decoder_stack(x)
        # #Pass through decoders  
        # for layer in self.decoder_stack:
        #     x                       = layer(x)
        
        #Pass through lm_head to get logits
        return self.lm_head(x), y

   
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


    def generate(self,prompt:list[int],n_tokens=128,temperature=.5):
        self.eval()

        with torch.no_grad():
            tokens  = prompt
        
            while len(tokens) - len(prompt) < n_tokens:

                logits          = self(torch.tensor(tokens[-self.n_positions:],device=self.devices[0],requires_grad=False).unsqueeze_(dim=0),None)[0][0,-1,:]
                distribution    = torch.nn.functional.softmax(logits/temperature,dim=-1)
                next_token      = torch.multinomial(distribution,1)
                tokens          = tokens + [next_token]
        self.train()
        return tokens

if __name__ == "__main__":

    n_embed     = 4 
    n_ff        = n_embed*2 
    n_heads     = n_embed//2
    bs          = 8 
    n_positions = 4
    n_vocab     = 32768

    #Create model
    lm          = LMSteinshark(n_embed=n_embed,n_heads=n_heads,n_positions=n_positions,n_vocab=n_vocab,n_ff=n_ff,dropout=.1)
    from dataset import TokenizedDataset
    import numpy 
    toks        = numpy.load("C:/data/nlp/tokens0.npy")
    aa          = TokenizedDataset(toks,n_positions)
    batch       = aa.sample(1,6,lm.devices)
    lm.forward(batch['input_ids'],batch['target_ids'])
