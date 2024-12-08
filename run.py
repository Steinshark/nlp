import torch.optim
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from tokenizers.implementations import ByteLevelBPETokenizer
from dataset import  TokenizedDataset, InfSampler, create_token_file, TextFileDataset
import torch    
from torch.utils.data import DataLoader
import time 
import os 
import random 
import numpy
import argparse
from torch.nn import CrossEntropyLoss
import torch
from matplotlib import pyplot as plt 
import json 

_KEYTOKENS      = []
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
T_TYPE      = torch.float

#import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
                                             resid_pdrop=.05,
                                             attn_pdrop=.05,
                                             embd_pdrop=.05,
                                             torch_dtype=T_TYPE,
                                             layer_norm_epsilon=1e-5,
                                             scale_attn_by_inverse_layer_idx=True
                                             
                                             )

        #Create the model
        super(GPTSteinshark,self).__init__(self.config)
        self.train_device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.train_device)
        self.tokenizer          = tokenizer
        self.n_positions        = input_size
        self.vocab_size         = vocab_size
        self.name               = name
        self.warmup             = True
        self.train_iters        = 0
        self.n_params           = sum([p.numel() for p in self.parameters()])

        self.metadata           = {"tok_trained_on":0,"epochs_complete":0,"iters_trained_on":0}


    def train_stein(self,   
              n_iter=2**17,
              bs=4,
              warmup_bs=16,
              lr=.0002,
              warmup_lr=.0001,
              wd=.01,
              warmup_steps=2048,
              sample_text='my cat is',
              accu_steps=16,
              temp=.85,
              ds_root=""
              ):

        #Data Vars
        self.train_bs                   = bs
        self.warmup_bs                  = warmup_bs
        self.accu_steps                 = accu_steps
        self.eff_bs                     = bs*accu_steps

        #Train Vars
        self.total_training_iters       = n_iter
        self.lr                         = lr
        self.warmup_lr                  = warmup_lr
        self.nominal_lr                 = lr
        self.wd                         = wd 
        self.warmup_steps               = warmup_steps
        self.warming_up                 = True
        self.continue_training          = True 

        #Telemetry vars
        self.cur_epoch                  = self.metadata['epochs_complete']

        self.samples_trained_on         = 0
        self.tokens_trained_on          = self.metadata['tok_trained_on']
        self.tokens_trained_on_epoch    = self.metadata['tok_trained_on']
        self.show_update_every_t        = 30
        self.sample_every_t             = 30* 60   
        self.start_timestamp            = time.time()
        self.last_update_timestamp      = time.time()
        self.last_sample_timestamp      = time.time()
        self.sample_text                = sample_text
        self.temp                       = temp

        #Loss Vars
        self.epoch_losses               = [[]] 
        self.batch_times                = [] 
        self.batch_tokens               = []
        self.batch_through              = []

        #Percent, lr, eff_bs
        self.schedule                   = [(.25,.00005,512)] 


        #Send model to device and prep for training 

        self.train(True).to(DEVICE)

        #Create the data pipeline1
        print(f"\tPreparing data")
        if not os.path.exists("C:/data/nlp/tokens.npy"):
            print(f"\tLoading Text Files")
            dataset                 = TextFileDataset(ds_root=ds_root,n_positions=self.n_positions)
            print(f"\tTokenizing")
            dataset.tokenize(self.tokenizer)
            np_arr:numpy.ndarray    = dataset.tokens
            np_arr                  = np_arr.astype(int)
            numpy.save("C:/data/nlp/tokens.npy",np_arr)
        else:
            np_arr                  = numpy.load("C:/data/nlp/tokens.npy")
            
        print("\tData Ready")

        #Convert dataset.tokens into a numpy array 
        dataset                 = TokenizedDataset(list(np_arr),n_positions=self.n_positions)
        self.dataloader         = DataLoader(dataset,batch_size=warmup_bs,sampler=InfSampler(),pin_memory=True).__iter__()
        self.tokens_trained_on_epoch    = self.metadata['tok_trained_on'] % dataset.n_tokens

        #Create the optimizer
        self.optimizer          = torch.optim.AdamW(self.parameters(),lr=warmup_lr,weight_decay=wd,betas=(.9,.95))
        #self.optimizer          = torch.optim.Adafactor(self.parameters(),lr=warmup_lr,weight_decay=self.wd)
        
        #Print training params 
        print(f"\tmodel params:\t{self.n_params//1_000_000:.2f}M params")
        print(f"\tdataset size:\t{len(dataset.tokens)/1_000_000:.1f}M tokens")
        print(f"\teff batchsize:\t{(self.accu_steps*self.train_bs)}")
        print(f"\twarmup lr:\t{self.warmup_lr}")
        print(f"\ttrain lr\t{self.lr}")
        print(f"\twarmup bs:\t{self.warmup_bs}")
        print(f"\ttrain bs\t{self.train_bs}\n\n\n")
        plt.ion()
        plt.show()


        #TRAINING LOOP
        cur_training_iter       = self.metadata['iters_trained_on']
    
        while cur_training_iter < self.total_training_iters:
                
                #get time 
                batch_start_t               = time.time()                

                #Zero Grad 
                for param in self.parameters():
                    param.grad              = None 


                #Update training state 
                self.update_training_state(cur_training_iter)

                #Synthetic BS increase
                batch_tokens_through                = 0 
                self.epoch_losses[-1].append(0)
                for _ in range(self.accu_steps):

                    #Get batch
                    batch                           = self.dataloader.__next__()


                    #Prep data
                    input_ids:torch.Tensor          = batch['input_ids'].to(DEVICE).type(torch.long)
                    num_tokens                      = input_ids.numel()
                    num_examples                    = input_ids.shape[0]

                        

                    #Send forward
                    lm_outputs                      = self(input_ids=input_ids,labels=input_ids,attention_mask=torch.ones(size=input_ids.shape,dtype=torch.long,device=torch.device('cuda')))
                    self.samples_trained_on         += num_examples
                    self.tokens_trained_on          += num_tokens
                    self.tokens_trained_on_epoch    += num_tokens
                    batch_tokens_through            += num_tokens
                    self.metadata['tok_trained_on'] += num_tokens


                    #Calc loss
                    loss                            = lm_outputs.loss / self.accu_steps

                    #Backpropogate the loss
                    loss.backward()
                    self.epoch_losses[-1][-1]       += loss

                #Clip norms
                torch.nn.utils.clip_grad_norm_(self.parameters(),numpy.pi)
                self.epoch_losses[-1][-1]           = self.epoch_losses[-1][-1].detach().cpu().item()
                self.metadata['iters_trained_on'] += 1

                #Step optimizer 
                self.optimizer.step()


                #Update telemetry variables
                self.batch_times.append(time.time()-batch_start_t)
                self.batch_through.append(batch_tokens_through/self.batch_times[-1])


                #Check for printing stats
                self.process_stats(cur_training_iter)
                cur_training_iter += 1

                
                plt.cla()
                plt.clf()
                plt.plot(self.epoch_losses[-1],label="Model Loss",color='dodgerblue')
                plt.title(f"Model Loss - {self.n_params//1_000_000}M params - [{cur_training_iter}/{self.total_training_iters}]")
                plt.legend()
                plt.draw()
                plt.pause(.01)


                


        plt.cla()
        plt.plot(self.warmup_losses,label="Warmup loss",color='dodgerblue')
        plt.plot([0 for _ in self.warmup_losses] + self.train_losses,label="Train loss",color='goldenrod')
        plt.title("Model Loss")
        plt.legend()
        plt.draw()
        plt.pause(.01)
        input(f"finish training")
    
    
    def train_stein_accel(self,   
              n_iter=2**17,
              bs=4,
              warmup_bs=16,
              lr=.0002,
              warmup_lr=.0001,
              wd=.01,
              warmup_steps=2048,
              sample_text='my cat is',
              accu_steps=16,
              temp=.85,
              ds_root=""
              ):

        #Data Vars
        self.train_bs                   = bs
        self.warmup_bs                  = warmup_bs
        self.accu_steps                 = accu_steps
        self.eff_bs                     = bs*accu_steps

        #Train Vars
        self.total_training_iters       = n_iter
        self.lr                         = lr
        self.warmup_lr                  = warmup_lr
        self.nominal_lr                 = lr
        self.wd                         = wd 
        self.warmup_steps               = warmup_steps
        self.warming_up                 = True
        self.continue_training          = True 

        #Telemetry vars
        self.cur_epoch                  = self.metadata['epochs_complete']

        self.samples_trained_on         = 0
        self.tokens_trained_on          = self.metadata['tok_trained_on']
        self.tokens_trained_on_epoch    = self.metadata['tok_trained_on']
        self.show_update_every_t        = 16
        self.sample_every_t             = 30* 60   
        self.start_timestamp            = time.time()
        self.last_update_timestamp      = time.time()
        self.last_sample_timestamp      = time.time()
        self.sample_text                = sample_text
        self.temp                       = temp

        #Loss Vars
        self.epoch_losses               = [[]] 
        self.batch_times                = [] 
        self.batch_tokens               = []
        self.batch_through              = []

        #Percent, lr, eff_bs
        self.schedule                   = [(.25,.0002,512)] 


        #Send model to device and prep for training 
        self.train(True)
        from accelerate import Accelerator
        accel           = Accelerator()

        #Create the data pipeline1
        print(f"\tPreparing data")
        if not os.path.exists("C:/data/nlp/tokens.npy"):
            print(f"\tLoading Text Files")
            dataset                 = TextFileDataset(ds_root=ds_root,n_positions=self.n_positions)
            print(f"\tTokenizing")
            dataset.tokenize(self.tokenizer)
            np_arr:numpy.ndarray    = dataset.tokens
            np_arr                  = np_arr.astype(int)
            numpy.save("C:/data/nlp/dataset.npy",np_arr)
        else:
            np_arr                  = numpy.load("C:/data/nlp/tokens.npy")
            
        print("\tData Ready")

        #Convert dataset.tokens into a numpy array 
        eval_set                      = 16384
        dataset_train                 = TokenizedDataset(list(np_arr[eval_set:]),n_positions=self.n_positions)
        dataset_test                  = TokenizedDataset(list(np_arr[eval_set:]),n_positions=self.n_positions)
        self.dataloader_train         = DataLoader(dataset_train,batch_size=train_bs,sampler=InfSampler(),pin_memory=True)
        #self.dataloader_test          = DataLoader(dataset_test,batch_size=train_bs,sampler=InfSampler(),pin_memory=True)
        self.tokens_trained_on_epoch    = self.metadata['tok_trained_on'] % dataset_train.n_tokens
        #Create the optimizer
        #self.optimizer          = torch.optim.AdamW(self.parameters(),lr=warmup_lr,weight_decay=wd,betas=(.9,.95))
        self.optimizer          = torch.optim.Adafactor(self.parameters(),lr=warmup_lr,weight_decay=self.wd)
        
        model, optimizer,self.train_dl     = accel.prepare(self,self.optimizer,self.dataloader_train,device_placement=[False,True,True])
        #Print training params 
        print(f"\tmodel params:\t{self.n_params//1_000_000:.2f}M params")
        print(f"\tdataset size:\t{len(dataset_train.tokens)/1_000_000:.1f}M tokens")
        print(f"\teff batchsize:\t{(self.accu_steps*self.train_bs)}")
        print(f"\twarmup lr:\t{self.warmup_lr}")
        print(f"\ttrain lr\t{self.lr}")
        print(f"\twarmup bs:\t{self.warmup_bs}")
        print(f"\ttrain bs\t{self.train_bs}\n\n\n")
        plt.ion()
        plt.show()


        #TRAINING LOOP
        cur_training_iter       = self.metadata['iters_trained_on']
    
        for batch in self.train_dl:
                
                #get time 
                batch_start_t               = time.time()                

                #Zero Grad 
                optimizer.zero_grad()


                #Update training state 
                self.update_training_state(cur_training_iter)

                #Synthetic BS increase
                batch_tokens_through                = 0 
                self.epoch_losses[-1].append(0)


                #Prep data
                #input_ids:torch.Tensor          = batch['input_ids'].to(DEVICE).type(torch.long)
                num_tokens                      = self.train_bs * self.n_positions
                num_examples                    = self.train_bs

                    

                #Send forward
                lm_outputs                      = model(**batch)
                self.samples_trained_on         += num_examples
                self.tokens_trained_on          += num_tokens
                self.tokens_trained_on_epoch    += num_tokens
                batch_tokens_through            += num_tokens
                self.metadata['tok_trained_on'] += num_tokens


                #Calc loss
                loss                            = lm_outputs.loss

                #Backpropogate the loss
                accel.backward(loss)
                self.epoch_losses[-1][-1]       += loss

                #Clip norms
                torch.nn.utils.clip_grad_norm_(self.parameters(),numpy.pi)
                self.epoch_losses[-1][-1]           = self.epoch_losses[-1][-1].detach().cpu().item()
                self.metadata['iters_trained_on'] += 1

                #Step optimizer 
                optimizer.step()


                #Update telemetry variables
                self.batch_times.append(time.time()-batch_start_t)
                self.batch_through.append(batch_tokens_through/self.batch_times[-1])


                #Check for printing stats
                self.process_stats(cur_training_iter)
                cur_training_iter += 1

                
                plt.cla()
                plt.clf()
                plt.plot(self.epoch_losses[-1],label="Model Loss",color='dodgerblue')
                plt.title(f"Model Loss - {self.n_params//1_000_000}M params - [{cur_training_iter}/{self.total_training_iters}]")
                plt.legend()
                plt.draw()
                plt.pause(.01)


                


        plt.cla()
        plt.plot(self.warmup_losses,label="Warmup loss",color='dodgerblue')
        plt.plot([0 for _ in self.warmup_losses] + self.train_losses,label="Train loss",color='goldenrod')
        plt.title("Model Loss")
        plt.legend()
        plt.draw()
        plt.pause(.01)
        input(f"finish training")



    def process_stats(self,cur_training_iter,save_every=200):
        
        #Check for model print stats
        if time.time() - self.last_update_timestamp > self.show_update_every_t:

            #Format and print 
            epoch       = f"[EP" + f"{self.cur_epoch}".rjust(2) + "]"
            perc        = f"{100*self.tokens_trained_on_epoch / self.dataloader._dataset.n_tokens:.1f}% through".rjust(14) 
            cur_iter    = f"{cur_training_iter}".rjust(8)
            iter        = f"iter: {cur_iter}/{self.total_training_iters//1_000}K".rjust(22)
            loss        = f"loss: {sum(self.epoch_losses[-1][-16:]) / len(self.epoch_losses[-1][-16:]):.4f}"
            tokens      = f"{self.tokens_trained_on/1_000_000:.1f}M tokens trained on".rjust(27)
            through     = f"{self.batch_through[-1]/1_000:.1f}K tok/s".rjust(15)

            print(f"{epoch} {perc}\t{iter}\t  {loss}{tokens}{through}")

            self.last_update_timestamp  = time.time()

        if time.time() - self.last_sample_timestamp > self.sample_every_t:
            
            #Use eval mode
            self.eval()

            #Req no grad
            with torch.no_grad():
                text                = sample_text
                encoded             = self.tokenizer.encode(text).ids

                while len(encoded) < 128 and not "<|endoftext|>" in text:
                    inputs              = torch.tensor(encoded).to(DEVICE)[-self.n_positions:]
                    mask                = torch.ones(len(encoded)).to(DEVICE)[-self.n_positions:]
                    logits              = self.forward(inputs,attention_mask=mask).logits[-1,:]
                    weights             = torch.nn.functional.softmax(logits/self.temp,dim=-1).detach().cpu()
                    choice              = random.choices(list(range(len(weights))),weights=weights,k=1)
                    text                += self.tokenizer.decode(choice)
                    encoded             = self.tokenizer.encode(text).ids

                text        = f'\n{text}'

            print(f"\n\n\nMODEL SAMPLE [{self.tokens_trained_on//1_000}k tokens]:{text}\n\n\n")
            
            
            self.train()
            self.last_sample_timestamp  = time.time()

        if cur_training_iter % save_every == 0:
            modelname   = f"stein_{self.metadata['iters_trained_on']-1}.pt"
            self.save(modelname)
            print(f"\n\n\nSaved {self.metadata['iters_trained_on']-1}k iter model\n\n\n")
 

    def update_training_state(self,cur_training_iter):

        if self.tokens_trained_on_epoch / self.dataloader._dataset.n_tokens > 1:
            self.tokens_trained_on_epoch = 0 
            self.cur_epoch += 1
            self.epoch_losses.append([])
            self.metadata['epochs_complete'] += 1
            #reset epoch 
            a = 1

        #Get percent of way through
        self.training_percent   = self.tokens_trained_on_epoch / self.dataloader._dataset.n_tokens

        #Check against 
        if self.schedule and self.schedule[0][0] < self.training_percent:
            
            #Pop it 
            setpoint,lr,bs      = self.schedule.pop(0)
            self.update_lr(lr)
            self.accu_steps     = bs // self.train_bs
            print(f"\n\nupdate lr to {lr:.5f} and bs to {bs}\n\n")


    
        #After warmup 
        if cur_training_iter < self.warmup_steps:
            self.warmup_lr             *= 1.016
            self.warmup_lr             = min(self.warmup_lr,self.lr)
            pass

        elif cur_training_iter >= self.warmup_steps and self.warming_up:
            
            print(f"\n\nSwitiching to non-warmup mode\n\n")
            self.dataloader._dataset.warmup = False
            #Set to non-warming up
            self.warming_up     = False 

            self.update_lr(self.lr)

    
    def update_lr(self,new_lr):
        #Switch to new optimizer 
        for pg in self.optimizer.param_groups:
            pg['lr']    = new_lr


    def generate(self,prompt):

        #Ensure model in eval mode and use no_grad context  
        self.eval()

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
                logits              = self.forward(tokenized_text,attention_mask=mask).logits[-1,:]

                #Sample next token from probabilities
                probs               = torch.nn.functional.softmax(logits,dim=-1).detach().cpu().numpy()
                choice              = random.choices(list(range(len(probs))),weights=probs,k=1)
                next_word           = self.tokenizer.decode(choice)

                #Add to text and encode once more
                print(f"{next_word}",end='',flush=True)
                text                += next_word
        
        print(f"'\n\n")


    def save(self,name,root="C:/data/nlp/models/"):

        savepath    = os.path.join(root,name)

        #Save metadata 
        with open(f"{os.path.join(root,'metadata'+name)}.json",'w') as writefile:
            writefile.write(json.dumps(self.metadata))
        
        #save params 
        torch.save(self.state_dict(),savepath)
    

    def load(self,name,root="C:/data/nlp/models/"):
        self.load_state_dict(torch.load(os.path.join(root,name),weights_only=True))

        with open(f"{os.path.join(root,'metadata'+name)}.json",'r') as readfile:
            self.metadata = json.loads(readfile.read())
        
        print(f"Loaded model {name}")

        

if __name__ == "__main__":
    
    argparser   = argparse.ArgumentParser()
    argparser.add_argument('--load_model',default='')
    argparser.add_argument('--load_vocab',default='True')
    argparser.add_argument('--train_dir',default='c:/data/nlp/train_dir')
    args        = argparser.parse_args()


    #Training/Model Settings 
    virtual_bs  = 64
    warmup_bs   = 4     
    train_bs    = 4     
    warmup_lr   = .00002
    train_lr    = .00005
    wd          = .01
    warmup_steps= 16
    input_size  = 1024
    vocab_size  = 32768
    embed_size  = 1024
    n_layers    = 16
    n_heads     = embed_size//64
    accu_steps  = virtual_bs//train_bs
    train_root  = args.train_dir
    sample_text = "hi guys! so i trained a large language ai model to develop scripts for me and here is the output:"
    
    
    print(f"loading vocab: {args.load_vocab}")
    print(f"loading model: {args.load_model}\n\n")


    #Create Tokenizer
    if not args.load_vocab == "True":
        print(f"Training tokenizer size={vocab_size}")
        tokenizer               = ByteLevelBPETokenizer()
        tokenizer.train([os.path.join(train_root,fname) for fname in os.listdir(train_root)],vocab_size=vocab_size)
        tokenizer.add_tokens(["<|endoftext|>"])
        if not os.path.exists("stein_tokenizer_bpe"):
            os.mkdir('stein_tokenizer_bpe')
        tokenizer.save_model('stein_tokenizer_bpe')
        print(f"\tcomplete")

    else:
        tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename="stein_tokenizer_bpe/vocab.json",merges_filename="stein_tokenizer_bpe/merges.txt")
    

    #Create model
    model                       = GPTSteinshark(input_size=input_size,vocab_size=vocab_size,n_embed=embed_size,n_layer=n_layers,n_head=n_heads,tokenizer=tokenizer,act_fn='gelu_new').to(DEVICE)
    model.name                  = "steinshark1.model"
   
    if args.load_model:
        mname   = args.load_model
        model.load(mname)

        # #Save each transformer then lm_head 
        # torch.save(model.transformer.state_dict(),"C:/data/nlp/xfmer.pt")
        # torch.save(model.lm_head.state_dict(),"C:/data/nlp/lmhead.pt")
        # exit()
    else:
        print(f"create new model: models/{model.name} ({model.n_params//1_000_000}M Params)")


    #Train
    model.train_stein(n_iter=16*1024,
                      warmup_bs=warmup_bs,
                      bs=train_bs,
                      warmup_lr=warmup_lr,
                      lr=train_lr,
                      wd=wd,
                      warmup_steps=warmup_steps,
                      sample_text=sample_text,
                      accu_steps=accu_steps,
                      temp=.99,
                      ds_root=train_root
                      )
    

    #Create local directory if it doesn't exist
    if not os.path.exists("models"):
        os.mkdir("models")
    
    #Save model
    torch.save(model.state_dict(),f"models/{model.name}")

    print(f"{model.model}")