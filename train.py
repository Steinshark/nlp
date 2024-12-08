import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from transformer import LMSteinshark
import os 
import argparse
from torch.utils.data import DataLoader
from dataset import TokenizedDataset, InfSampler
import numpy 
import time 
from matplotlib import pyplot as plt 
import sys 
sys.path.append("C:/gitrepos/steinpy/src")
from steinpy.utils import reduce_arr 

_UPDATE_EVERY_T                 = 30
_SAMPLE_EVERY_T                 = 5*60
_LAST_UPDATE_T                  = time.time() 
_LAST_SAMPLE_T                  = time.time() - (4*60)
_SAVE_MODEL_EVERY               = 250
_N_TOKENS                       = 860308519

def print_update(current_step,total_step,losses,tok_thruput,model:LMSteinshark,prompt:str,tokenizer:ByteLevelBPETokenizer,optimizer:torch.optim.Adam):
    global _LAST_UPDATE_T
    global _LAST_SAMPLE_T

    #Check for printint stats 
    if time.time() - _LAST_UPDATE_T > _UPDATE_EVERY_T:

        iters                   = "iter " + f"{current_step}/{total_step}".rjust(11) + "   "
        losses                  = f"{float(sum(losses[-64:])) / float(len(losses[-64:])):.5f}".rjust(8) + "   "
        tok_thru                = f"{tok_thruput/1_000:.1f}k tok/s" + "   "
        toks                    = f"{model.stats['tok_through']/1_000_000:.1f}M tokens"
        lr                      = f"  lr={optimizer.param_groups[0]['lr']}"
        _LAST_UPDATE_T          = time.time()

        print(iters+losses+tok_thru+toks+lr)

    #Check to save model 
    if current_step % _SAVE_MODEL_EVERY == 0 and not current_step == 0:
        model.save()

    #Check to sample 
    if time.time() - _LAST_SAMPLE_T > _SAMPLE_EVERY_T:
        print(f"\n\nGenerating:")
        print(f"{tokenizer.decode(model.generate(tokenizer.encode(prompt).ids,n_tokens=128))}\n\n")
        _LAST_SAMPLE_T          = time.time()

if __name__ == "__main__":

    #Ensure optimizations 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    #Handle arguments
    argparser                   = argparse.ArgumentParser()
    argparser.add_argument('--model_dir',default='')
    argparser.add_argument('--load_vocab',default='True')
    argparser.add_argument('--train_dir',default='c:/data/nlp/train_dir')
    args                        = argparser.parse_args()


    #Training/Model Settings 
    train_batch_tok             = 768 * 1024
    bs                          = 30
    lr                          = .0001
    wd                          = 0
    warmup_steps                = 20
    input_size                  = 512
    vocab_size                  = 50_000
    embed_size                  = 1200
    n_layers                    = 24
    dropout                     = .1
    n_heads                     = embed_size//100
    act_fn                      = torch.nn.GELU
    n_ff                        = int(embed_size*3)
    train_root                  = args.train_dir
    tok_trained_on              = 0 
    virtual_bs                  = train_batch_tok // (bs*input_size)
    accu_steps                  = virtual_bs // bs
    pct_start                   = .2
    train_iters                 = 2*_N_TOKENS // (bs*input_size)
    lr_steps                    = 2*_N_TOKENS // virtual_bs
    trig_embd                   = False
    sample_text                 = "hi guys! welcome to my python tutorial, today we're going to learn about"

    #Create Tokenizer
    if not args.load_vocab == "True":
        print(f"Training tokenizer size={vocab_size}")
        tokenizer               = ByteLevelBPETokenizer()
        tokenizer.train([os.path.join(train_root,fname) for fname in os.listdir(train_root)],vocab_size=vocab_size-1)
        tokenizer.add_tokens(["<|endoftext|>"])
        if not os.path.exists("stein_tokenizer_bpe"):
            os.mkdir('stein_tokenizer_bpe')
        tokenizer.save_model('stein_tokenizer_bpe')
        print(f"\tcomplete")

    else:
        tokenizer               = ByteLevelBPETokenizer().from_file(vocab_filename="stein_tokenizer_bpe/vocab.json",merges_filename="stein_tokenizer_bpe/merges.txt")
    

    #Create model 
    model                       = LMSteinshark(input_size,embed_size,n_layers,n_heads,n_ff,vocab_size,act_fn,dropout=dropout,trig_embd=trig_embd)
    model                       = model.bfloat16()
    #model.load()
    #Create optimizer 
    optimizer                   = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=wd)
    lr_sched                    = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,pct_start=.2,total_steps=lr_steps)

    #Create loaders 
    tokens                      = numpy.load("C:/data/nlp/tokens.npy").astype(numpy.int16)
    dataset                     = TokenizedDataset(tokens,n_positions=input_size)
    # inputs_alloc_tensor         = torch.empty(size=(bs,input_size),dtype=torch.long,device=model.device)
    # targets_alloc_tensor        = torch.empty(size=(bs,input_size),dtype=torch.long,device=model.device)
    #Create updates 
    losses                      = [] 
    tok_thru_per_iter           = []
    time_per_iter               = [] 

    #Train model 
    cur_train_iter              = 0 

    print(f"Beginning training\n\tModel Size:\t{model.n_params//1_000_000}M params\n\tData Size:\t{dataset.n_tokens//1_000_000}M Tokens\n\tEmbeddings:\t{'Trig' if trig_embd else 'Learned'}\n")
    plt.ion()
    plt.show()


    while cur_train_iter < train_iters:

        t0                          = time.time()

        #Load data
        batch                       = dataset.sample(bs,input_size,model.devices)
        input_ids                   = batch['input_ids']
        target_ids                  = batch['target_ids']

        #Put through model 
       # with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        logits                      = model(input_ids).view(bs*input_size,vocab_size)
        targets                     = target_ids.view(bs*input_size)

        #Compute and backward loss 
        loss                        = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward() 
        model.stats['tok_through']  += float(input_ids.numel())
        model.stats['iter_through'] += 1
        model.stats['losses'].append(float(loss.detach()))

        tok_thru_per_iter.append(input_ids.numel())
        time_per_iter.append(time.time()-t0)

        #Zero if on step cycle 
        if cur_train_iter % accu_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),2)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()

            plt.cla()
            plt.clf()
            plt.plot(reduce_arr(model.stats['losses'],256),label="Model Loss",color='dodgerblue')
            plt.title(f"Model Loss - {model.n_params//1_000_000}M params - [{cur_train_iter}/{train_iters}]")
            plt.legend()
            plt.draw()
            plt.pause(.001)
        
        #Check if new epoch 
        if model.stats['tok_through'] // dataset.n_tokens > model.stats["eps_through"]:
            model.stats["eps_through"] += 1

        print_update(cur_train_iter,train_iters,model.stats['losses'],sum(tok_thru_per_iter[-8:])/len(tok_thru_per_iter[-8:]),model,sample_text,tokenizer,optimizer)

        cur_train_iter += 1 
    
    input("training complete")


