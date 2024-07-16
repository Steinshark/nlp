from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast, Trainer,BasicTokenizer
import torch    
from torch.utils.data import DataLoader, Dataset
import json 
import os 
import random 
from transformers import GPT2Model,GPT2Config,GPT2Tokenizer,GPT2LMHeadModel,Trainer,TrainingArguments, DataCollatorForLanguageModeling, GenerationConfig
import math 


DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class GPTSteinsharkTokenizer():

    def __init__(self,vocabulary: str|list|dict):
        self.base_chars     = set()

        if isinstance(vocabulary,str):
            with open(vocabulary,"r",encoding='utf_8') as file:
                self.words     = {word:i for i,word in enumerate(json.loads(file.read()))}
                self.tokens    = {value:key for key,value in self.words.items()}

        elif isinstance(vocabulary,list):
            self.tokens     = {i:item for i,item in enumerate(vocabulary)}
        elif isinstance(vocabulary,dict):
            self.tokens     = vocabulary
        

        self.pad_token  = '<|PAD|>'
        
    
    def munch(self,candidate:str,text:str):
        candidate   += text[0]
        text        = text[1:]

        return candidate, text 


    def peek(self,candidate:str,text:str):

        if not text:
            return False 
        
        next_candidate  = candidate + text[0]

        #print(f"break on {next_candidate} in words: {next_candidate in self.words}")
        return next_candidate in self.words
    

    def encode(self,text:str,buffer_to:int=0,prediction:bool=False):
        encoding    = [] 
        original    = ''


        #Find largest token that fits:

        while text:
            
            top_token   = ''
            top_n       = 0
            #Go through all tokens
            for token in self.words:
                token_len   = len(token)

                if len(text) >= token_len:
                    if text[:token_len] == token and token_len > top_n:
                        top_n       = token_len
                        top_token   = token 
            
            #add token and remove from text 
            #print(f"found token {top_token} [{top_token in self.words}]\nlen={len(encoding)}")
            encoding.append(self.words[top_token])
            text        = text[top_n:]
        
        if buffer_to:
            encoding += [self.words[self.pad_token]] * (buffer_to - len(encoding))


        unpadded_context        = [t for t in encoding]
        padded_context          = None 
        prediction_token        = None 
        padded_mask             = None 
        unpadded_mask           = None

        if prediction:
            prediction_token    = [encoding[-1]]
            unpadded_context    = unpadded_context[:-1]
            unpadded_mask       = [0 if tok == self.words[self.pad_token] else 1 for tok in unpadded_context]

        if buffer_to:
            padded_context      = unpadded_context + [self.words[self.pad_token]] * (buffer_to - len(unpadded_context))
            padded_mask         = [0 if tok == self.words[self.pad_token] else 1 for tok in padded_context]
        
        #Build mask 
        return {"unpadded_context":unpadded_context,"padded_context":padded_context,"padded_mask":padded_mask,"unpadded_mask":unpadded_mask}



    def encode_n(self,text:str,n:int,buffer_to:int=0,prediction:bool=True):
        encoding    = [] 
        original    = ''


        #Find largest token that fits:

        while len(encoding) < n:
            
            top_token   = ''
            top_n       = 0
            #Go through all tokens
            for token in self.words:
                token_len   = len(token)

                if len(text) >= token_len:
                    if text[:token_len] == token and token_len > top_n:
                        top_n       = token_len
                        top_token   = token 
            
            #add token and remove from text 
            encoding.append(self.words[top_token])
            text        = text[top_n:]
        

        unpadded_context        = [t for t in encoding]
        padded_context          = None 
        next_token              = None 
        padded_mask             = None 
        unpadded_mask           = None

        if prediction:
            next_token          = unpadded_context[-1]
            unpadded_context    = unpadded_context[:-1]
            unpadded_mask       = [0 if tok == self.words[self.pad_token] else 1 for tok in unpadded_context]

        if buffer_to:
            padded_context      = unpadded_context + [self.words[self.pad_token]] * (buffer_to - len(unpadded_context))
            padded_mask         = [0 if tok == self.words[self.pad_token] else 1 for tok in padded_context]
        
        #Build mask 
        return {"unpadded_context":unpadded_context,"padded_context":padded_context,"padded_mask":padded_mask,"unpadded_mask":unpadded_mask,"next_token":next_token}


    def decode(self,tokens:list[int]):
        return ''.join([self.tokens[token] for token in tokens])

        

class GPTSteinsharkDataSet(Dataset):

    def __init__(self,ds_root:str,tokenizer:GPT2Tokenizer,n_positions:int,is_dir=True):

        #Load all files into as list of texts
        if is_dir:
            self.filenames      = [os.path.join(ds_root,file) for file in os.listdir(ds_root)]  
        else:
            self.filenames      = [ds_root]
        self.texts          = [open(fname,'r',encoding='utf_8').read().lower() for fname in self.filenames]

        #Create one string of text
        text                = ''
        for file in self.texts:
            text += (file + tokenizer.eos_token) 
        self.texts          = text

        #Set class variables
        self.tokenizer      = tokenizer
        self.n_positions    = n_positions
        self.select_pos     = n_positions
        self.eos_token      = tokenizer.eos_token
        self.pad_token      = tokenizer.pad_token
        self.size           = 1024 * 64

    #returns text_ids and attention_mask
    def __getitem__(self,i):
        
        #Pick random text 
        text                = self.texts

        try:
            #Pick random index 
            start_i             = random.randint(0,len(text)-self.select_pos)
        

            #Pick random length 
            encodings           = self.tokenizer(text[start_i:start_i+1_000],return_tensors='pt',padding=True)

            if len(encodings['input_ids'][0]) < self.n_positions:

                #Convert to lists
                encodings['input_ids']              = encodings['input_ids'][0].tolist()
                encodings['attention_mask']         = encodings['attention_mask'][0].tolist()
                
                #Pad 
                encodings['input_ids']              += [self.tokenizer.pad_token_id] * (self.n_positions - len(encodings['input_ids'])) 
                encodings['attention_mask']         += [0] * (self.n_positions - len(encodings['attention_mask'])) 

                #Convert back to tensors
                encodings['input_ids']              = torch.tensor(encodings['input_ids'])
                encodings['attention_mask']         = torch.tensor(encodings['attention_mask'])

                return encodings
            else:

                returndict          = {key:value[0][:self.n_positions] for key,value in encodings.items()}

            return returndict

        except ValueError as v:
            print(v)
            return self.__getitem__(0)
         



    def __len__(self):
        return self.size


    def get_iter(self,max_i=100_000_000):
        i   = 0 

        text_files          = [open(fname,'r',encoding='utf_8').read().lower() for fname in self.filenames]
        print(f"number of text files: {len(text_files)}")
        for text in text_files:
            i += 1 
            if i > max_i:
                break
            yield text


    def save_to_file(self,fname:str):
        with open(fname,'w',encoding='utf_8') as file:
            file.write(self.texts)
        return 



class GPTSteinshark(GPT2LMHeadModel):

    def __init__(self,
                 input_size=1024,
                 vocab_size=32768,
                 n_embed=768,
                 n_layer=12,
                 n_head=12,
                 act_fn="gelu_new",
                 name="steinshark1"

                 ):
        
        #Create config for the model 
        self.config             = GPT2Config(vocab_size=vocab_size,
                                             n_positions=input_size,
                                             n_embd=n_embed,
                                             n_layer=n_layer,
                                             n_head=n_head,
                                             activation_function=act_fn,
                                             )

        #Create the model
        super(GPTSteinshark,self).__init__(self.config)
        self.train_device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model              = GPT2LMHeadModel(self.config).to(self.train_device)
        self.n_positions        = input_size
        self.vocab_size         = vocab_size
        self.name               = name


    def train_stein(self,
              tokenizer:GPTSteinsharkTokenizer,
              ds_root:str='alldata',
              n_iter=2**17,
              bs=4,
              lr=.0002,
              wd=.005,
              n_pred=8,
              clipping=True,
              n_generate=4,
              sample_text='my cat is',
              save_every=1024*8
              ):

        #Send model to device and prep for training 
        self.model      = self.model.to(DEVICE)
        self.model.float()
        self.model.train(True)

        #Create the data pipeline 
        dataloader              = GPTSteinsharkDataSet(ds_root=ds_root,tokenizer=tokenizer,n_positions=self.n_positions)
        dataloader      = DataLoader(dataloader,batch_size=bs,shuffle=True).__iter__()

        #Create the learning pipeline
        self.optimizer  = torch.optim.AdamW(self.model.parameters(),lr=lr,weight_decay=wd)

        #Track loss 
        losses          = [] 
        
        #Clip as specified
        if clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),math.pi/math.e)  #Literally no reason for this


        #Run the training n_iter times 
        for iter in range(n_iter):

            #Printout
            if iter % n_generate == 0:
                print(f"iter [{iter}]\t",end='')
            
            #Zero 
            self.optimizer.zero_grad()

            #Snatch a random item from the dataset
            try:
                training_data           = dataloader.__next__()
            except StopIteration:
                dataloader              = GPTSteinsharkDataSet(ds_root=ds_root,tokenizer=tokenizer,n_positions=self.n_positions)
                dataloader              = DataLoader(dataloader,batch_size=bs,shuffle=True).__iter__()
                training_data           = dataloader.__next__()

            #Prep data
            training_tokens         = training_data['input_ids'].to(self.train_device).type(torch.long).to(DEVICE)
            training_mask           = training_data['attention_mask'].to(self.train_device).type(torch.float32).to(DEVICE)
           
            #Send forward
            next_prediction         = self.model.forward(input_ids=training_tokens,attention_mask=training_mask,labels=training_tokens)

            #Send backward
            loss                    = next_prediction.loss
            loss.backward()

            #Track loss
            losses.append(loss.mean().item())

            #Optimize net
            self.optimizer.step()

            #Sample per 
            if iter % n_generate == 0:
                with torch.no_grad():
                    text    = sample_text
                    for _ in range(n_pred):
                        encoded             = tokenizer.encode(text)
                        inputs              = torch.tensor(encoded).to(DEVICE)[-self.n_positions:]
                        mask                = torch.ones(len(encoded)).to(DEVICE)[-self.n_positions:]
                        logits              = self.model.forward(inputs,attention_mask=mask).logits[-1,:]
                        probs               = torch.nn.functional.softmax(logits,dim=-1).detach().cpu().numpy()
                        choice              = random.choices(list(range(len(probs))),weights=probs,k=1)
                        chosen              = tokenizer.decode(choice)
                        text                = text + chosen

                text        = f'\n{text}'.replace(f"\n","\n\t")
                print(f"loss={losses[-1]:.4f}\n{text}\n\n\n\n")
            
            if iter % save_every == 0:
                #Create local directory if it doesn't exist
                if not os.path.exists("models"):
                    os.mkdir("models")
                torch.save(self.state_dict(),f"models/{self.name}")
                print(f"Saved model at iter{iter} to models/{self.name}")
            

    def test_ground(self,tokenizer:GPT2Tokenizer):
        text        = "This is a sample text that is rather not long. Will it work?"

    



if __name__ == "__main__":
    
    #Training/Model Settings 
    train_bs    = 8
    lr          = 1e-5
    input_size  = 128+64
    vocab_size  = 4096
    embed_size  = 1024
    n_layers    = 16 
    n_heads     = 32
    train_root  = 'alldata'
    sample_text = 'i live with my little family. we live in virginia. we have a little cat. let me tell you about him. '

    #Create Tokenizer
    build_tokenizer         = False
    if build_tokenizer:
        tokenizer               = AutoTokenizer.from_pretrained('gpt2')
        place_iter              = dataloader              = GPTSteinsharkDataSet(ds_root=train_root,tokenizer=tokenizer,n_positions=input_size).get_iter(max_i=15_000_000)
        tokenizer               = tokenizer.train_new_from_iterator(place_iter,vocab_size=vocab_size)
        tokenizer.save_pretrained("tokenizer2")
    else:
        tokenizer              = GPT2TokenizerFast.from_pretrained("tokenizer2")
    tokenizer.pad_token     = "<|pad|>" 

    #Build the model 
    loading     = False

    model       = GPTSteinshark(input_size=input_size,vocab_size=vocab_size,n_embed=embed_size,n_layer=n_layers,n_head=n_heads).to(DEVICE)
    model.name  = "steinshark1.model"
    if loading:
        print(f"loading model from models/{model.name}")
        model.model.load_state_dict(torch.load(f"models/{model.name}"))

    #Test tokenization on a large string 
    testing     = False 
    if testing:
        sample_string   = '''this is a sample string that is very long and contains complex vocabulary in order to test the tokenizer. It is multiple sentences long and contains a lot of words, sentences, and punctuation. It is a very long string that is used to test the tokenizer. Nuances must be present, as well as puncation. should we keep going? I think so. It must test at least a trained corpus and include crazy interesteing things that don't  appear anywhere else naturally or otherwise. Maintain some semblance of sanity, but not too much. Usurp the throne of the king of the world. Why? Because we can. We must. We will. We are. We are the world. We are the children. We are the ones who make a brighter day. So let's start giving. On another note, here is some sample shakespear: To be or not to be, that is the question. Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles and by opposing end them. To die, to sleep, no more, and by a sleep to say we end the heartache and the thousand natural shocks that flesh is heir to. Tis a consummation devoutly to be wished. To die, to sleep, to sleep perchance.'''
        print(tokenizer.decode(tokenizer.encode(sample_string)[:input_size]))
    

    #Train
    model.train_stein(tokenizer,train_root,n_iter=1024*256,bs=train_bs,lr=lr,n_pred=input_size-len(tokenizer.encode(sample_text)),n_generate=256,sample_text=sample_text,save_every=1024)

    #save model to models directory
    
    #Create local directory if it doesn't exist
    if not os.path.exists("models"):
        os.mkdir("models")
    
    #Save model
    torch.save(model.state_dict(),f"models/{model.name}")

    print(f"{model.model}")
