import torch.optim
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
from tokenizer import SteinTokenizer
from transformers import GPT2TokenizerFast
from tokenizers.implementations import ByteLevelBPETokenizer
from dataset import GPTSteinsharkDataSet
import torch    
from torch.utils.data import DataLoader
import json 
import os 
import random 
import math 
import argparse
from torch.nn import CrossEntropyLoss
import torch
from matplotlib import pyplot as plt 

_KEYTOKENS  = []

def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    #return loss_per_sample.mean()
    # Calculate and scale weighting
    weights = torch.stack([(inputs == kt).type(torch.float) for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )
    weights = alpha * (1.0 + weights)
    # Calculate weighted average
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss


DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T_TYPE      = torch.bfloat16


class GPTSteinshark(GPT2LMHeadModel):
    

    def __init__(self,
                 input_size=64,
                 vocab_size=1024,
                 n_embed=64,
                 n_layer=16,
                 n_head=4,
                 act_fn="gelu_new",
                 name="steinshark1",
                 tokenizer=ByteLevelBPETokenizer
                 ):
        
        #Create config for the model 
        self.config             = GPT2Config(vocab_size=vocab_size,
                                             n_positions=input_size,
                                             n_embd=n_embed,
                                             n_layer=n_layer,
                                             n_head=n_head,
                                             activation_function=act_fn,
                                             resid_pdrop=.1,
                                             torch_dtype=T_TYPE,
                                             layer_norm_epsilon=1e-5)

        #Create the model
        super(GPTSteinshark,self).__init__(self.config)
        self.train_device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model              = GPT2LMHeadModel(self.config).to(self.train_device)
        self.tokenizer          = tokenizer
        self.n_positions        = input_size
        self.vocab_size         = vocab_size
        self.name               = name
        self.warmup             = True
        self.train_iters        = 0


    def train_stein(self,
              ds_root:str='alldata',
              n_iter=2**17,
              bs=4,
              warmup_bs=16,
              lr=.0002,
              warmup_lr=.0001,
              wd=.01,
              warmup_steps=2048,
              abbrev_steps=32768,
              sample_text='my cat is',
              grad_accumulation_steps=16,
              ):

        #keep args
        self.n_iter             = n_iter
        self.lr                 = lr
        self.warmup_lr          = warmup_lr
        self.train_bs           = bs
        self.warmup_bs          = warmup_bs
        self.nominal_lr         = lr
        self.wd                 = wd 
        self.warmup_steps       = warmup_steps
        self.warming_up         = True
        self.ds_root            = ds_root
        self.train_iters        = n_iter
        self.samples_trained    = 0
        self.grad_accumulation_steps    = grad_accumulation_steps
        self.abbrev_steps       = abbrev_steps
        self.bs                 = bs
        #Send model to device and prep for training 
        self.model              = self.model.to(DEVICE)
        self.model.train(True)

        #Create the data pipeline 
        dataset                 = GPTSteinsharkDataSet(ds_root=ds_root,n_positions=self.n_positions)
        self.dataloader         = DataLoader(dataset,batch_size=warmup_bs,shuffle=True).__iter__()
        dataset.tokenized_text  = self.tokenizer.encode(dataset.text).ids
        dataset.print_stats()

        #Create the optimizer
        self.optimizer          = torch.optim.AdamW(self.parameters(),lr=warmup_lr,weight_decay=wd,betas=(.96,.9999))

        #Track loss 
        self.warmup_losses      = [] 
        self.train_losses       = [] 
        visited                 = set()

        #Print training params 
        print(f"\tparams:\t\t{sum([p.numel() for p in self.model.parameters()])//1_000_000:.2f}M params")
        print(f"\ttrainset:\t{len(dataset.tokenized_text)/1_000_000:.1f}M tokens")
        print(f"\twarmup steps:\t{self.warmup_steps}")
        print(f"\twarmup lr:\t{self.warmup_lr}")
        print(f"\ttrain lr\t{self.lr}")
        print(f"\twarmup bs:\t{self.warmup_bs}")
        print(f"\ttrain bs\t{self.train_bs}")
        plt.ion()
        plt.show()

        #Run the training n_iter times 
        self.iter                       = -1
        self.accumulate_flag            = False
        while self.samples_trained < self.n_iter:

            for i, batch in enumerate(self.dataloader):
                if self.samples_trained > self.n_iter:
                    break
                #Update training parameters
                self.iter                   += 1
                self.update_training_params()
                self.accumulate_flag        = False

                #Prep data 
                tokens                      = torch.stack(batch)
                tokens                      = tokens.to(DEVICE).type(torch.long).t()#100% unsure why this needs to happen.... bad looks
                               
                #Send forward
                next_prediction             = self.model(tokens,labels=tokens)#attention_mask=masks
                self.samples_trained        += next_prediction.logits.shape[0]

                #Gather loss
                loss                        = next_prediction.loss #keytoken_weighted_loss(tokens,next_prediction.logits,_KEYTOKENS) #next_prediction.loss 
                loss                        = loss / (self.grad_accumulation_steps if not self.warming_up else 1)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)

                #Send back
                loss.backward()

                #Track loss
                if self.warming_up:
                    self.warmup_losses.append(loss.mean().item())
                else:
                    self.train_losses.append(loss.mean().item()*self.grad_accumulation_steps)

                #Optimize net over grad accum steps
                if self.samples_trained % self.grad_accumulation_steps == 0 or self.warming_up:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.accumulate_flag        = True
                    plt.cla()
                    plt.plot(self.warmup_losses,label="Warmup loss",color='dodgerblue')
                    plt.plot([0 for _ in self.warmup_losses] + self.train_losses,label="Train loss",color='goldenrod')
                    plt.title(f"Model Loss - {sum([p.numel() for p in self.model.parameters()])/1_000_000:.2f}M params - [{self.samples_trained}/{self.n_iter}]")
                    plt.legend()
                    plt.draw()
                    plt.pause(.01)


                #Sample every .1%
                if self.samples_trained % 1024 == 0:
                    self.model.eval()
                    print(f"iter [{self.samples_trained}/{self.n_iter}]\tgrad_acc [{self.grad_accumulation_steps}]")
                    text = ''
                    if self.samples_trained % 2048 == 0:
                        with torch.no_grad():
                            text                = sample_text
                            encoded             = self.tokenizer.encode(text).ids

                            #Produce the max number of tokens without giving up context
                            while len(encoded) < self.n_positions and not "<|endoftext|>" in text:
                                inputs              = torch.tensor(encoded).to(DEVICE)[-self.n_positions:]
                                mask                = torch.ones(len(encoded)).to(DEVICE)[-self.n_positions:]
                                logits              = self.model.forward(inputs,attention_mask=mask).logits[-1,:]
                                probs               = torch.nn.functional.softmax(logits,dim=-1).detach().cpu().numpy()
                                choice              = random.choices(list(range(len(probs))),weights=probs,k=1)
                                text                = text + self.tokenizer.decode(choice)
                                encoded             = self.tokenizer.encode(text).ids

                        text        = f'\n{text}'
                    print(f"loss={(self.warmup_losses[-1] if self.warming_up else self.train_losses[-1]):.4f}\tlr={self.lr if not self.warming_up else self.warmup_lr:.5f}\tprogress={100*self.training_percent:.1f}%\n{text}")
                    self.model.train(True)
                    visited.add(int(1000*self.training_percent))
                
                    #Create local directory if it doesn't exist and save checkpoint
                    if self.samples_trained % 16384 == 0:
                        if not os.path.exists("models"):
                            os.mkdir("models")
                        torch.save(self.state_dict(),f"models/{self.name}")
                        print(f"Saved model at iter{self.iter} to models/{self.name}\n\n\n\n")


        plt.cla()
        plt.plot(self.warmup_losses,label="Warmup loss",color='dodgerblue')
        plt.plot([0 for _ in self.warmup_losses] + self.train_losses,label="Train loss",color='goldenrod')
        plt.title("Model Loss")
        plt.legend()
        plt.draw()
        plt.pause(.01)
        input(f"finish training")


    def update_training_params(self):

        #Get percent of way through
        self.training_percent   = self.samples_trained / self.train_iters

    
        #After warmup 
        if self.samples_trained < self.warmup_steps:
            self.warmup_lr             *= 1.016
            self.warmup_lr             = min(self.warmup_lr,self.lr)
            pass

        elif self.samples_trained >= self.warmup_steps and self.warming_up:
            
            print(f"Switiching to non-warmup mode\n\n\n")
            #Set to non-warming up
            self.warming_up     = False 

            #Switch to new optimizer 
            self.optimizer.param_groups[0]['lr']    = self.lr
            #self.optimizer      = torch.optim.AdamW(self.model.parameters(),lr=self.lr,betas=(.9,.999),amsgrad=True)
            #self.optimizer          = torch.optim.SGD(self.parameters(),lr=self.lr,weight_decay=wd,momentum=.5)
            #Rebuild dataset with new bs 
            dataset             = GPTSteinsharkDataSet(ds_root=self.ds_root,n_positions=self.n_positions)
            dataset.warmup      = False
            self.dataloader     = DataLoader(dataset,batch_size=self.train_bs,shuffle=True).__iter__()

            #Show losses
            plt.plot(self.warmup_losses)
            #plt.show()
        
        elif self.accumulate_flag:
            pass
            
        if self.samples_trained >= self.abbrev_steps and self.dataloader._dataset.train_i < 1:
            print(f"\n\tTrain on full dataset")
            self.dataloader._dataset.train_i    = 1.
        

    def generate(self,prompt):

        #Ensure model in eval mode and use no_grad context  
        self.model.eval()

        with torch.no_grad():
        
            
            text            = prompt.lower()
            tokenized_text  = "placeholder"
            print(f"'")
            #Produce the max number of tokens without giving up context
            while len(tokenized_text) < self.n_positions and not self.tokenizer.end_token in text:
                #Convert tokenized inputs to tensor on proper device
                encoded_text        = self.tokenizer.encode(text)
                tokenized_text      = torch.tensor(encoded_text).to(DEVICE)
                mask                = torch.ones(len(tokenized_text)).to(DEVICE)

                #Send on forward pass
                logits              = self.model.forward(tokenized_text,attention_mask=mask).logits[-1,:]

                #Sample next token from probabilities
                probs               = torch.nn.functional.softmax(logits,dim=-1).detach().cpu().numpy()
                choice              = random.choices(list(range(len(probs))),weights=probs,k=1)
                next_word           = self.tokenizer.decode(choice)

                #Add to text and encode once more
                print(f"{next_word}",end='',flush=True)
                text                += next_word
        
        print(f"'\n\n")

    



if __name__ == "__main__":
    
    
    argparser   = argparse.ArgumentParser()
    argparser.add_argument('--load_model',default='True')
    argparser.add_argument('--load_vocab',default='True')
    argparser.add_argument('--train_dir',default='C:/gitrepos/nlp/yt_ascii')
    args    = argparser.parse_args()


    #Training/Model Settings 
    warmup_bs   = 16
    train_bs    = 16
    warmup_lr   = .0000002
    train_lr    = .00002
    wd          = .01
    warmup_steps= 2048
    input_size  = 32
    vocab_size  = 8192
    embed_size  = 256+128
    n_layers    = 8
    n_heads     = 12
    batch_mult  = 64
    train_root  = args.train_dir
    sample_text = 'welcome back to practical python. today we are going to learn about functions and how they work. well start'
    

    print(f"loading vocab: {args.load_vocab == 'True'}")
    print(f"loading model: {args.load_model == 'True'}")


    #Create Tokenizer
    if not args.load_vocab == "True":
        tokenizer               = ByteLevelBPETokenizer()
        tokenizer.train([os.path.join(train_root,fname) for fname in os.listdir(train_root)],vocab_size=vocab_size)
        if not os.path.exists("stein_tokenizer_bpe"):
            os.mkdir('stein_tokenizer_bpe')
        tokenizer.save_model('stein_tokenizer_bpe')
    else:
        tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename="stein_tokenizer_bpe/vocab.json",merges_filename="stein_tokenizer_bpe/merges.txt")
    
    #Make keytokens 
    for keyword in ["list","data","define","javascript","object","class","statement","variable","html","rust","python","coding","code","project","video","guys",'learn','function',"method","model","developer"]:
        keytoken    = tokenizer.encode(keyword).ids[0]
        _KEYTOKENS.append(keytoken)

    #Create model
    model                       = GPTSteinshark(input_size=input_size,vocab_size=vocab_size,n_embed=embed_size,n_layer=n_layers,n_head=n_heads,tokenizer=tokenizer,act_fn='gelu_new').to(DEVICE)
    model.name                  = "steinshark1.model"
   
    if args.load_model == "True":
        print(f"loading model from: models/{model.name}")
        model.load_state_dict(torch.load(f"models/{model.name}"))
    else:
        print(f"create new model: models/{model.name}")


    #Train
    model.train_stein(train_root,
                      n_iter=128*1024,
                      warmup_bs=warmup_bs,
                      bs=train_bs,
                      warmup_lr=warmup_lr,
                      lr=train_lr,
                      wd=wd,
                      warmup_steps=warmup_steps,
                      sample_text=sample_text,
                      grad_accumulation_steps=batch_mult)
    

    #Create local directory if it doesn't exist
    if not os.path.exists("models"):
        os.mkdir("models")
    
    #Save model
    torch.save(model.state_dict(),f"models/{model.name}")

    print(f"{model.model}")