import os 
import json

class SteinTokenizer:

    def __init__(self,text):

        self.text       = text
        self.eos_token  = '<|endoftext|>'
        self.pad_token  = '<|pad|>' 


    def train(self,vocab_size):

        working_text    = [letter for letter in self.text] + [self.eos_token]
        vocab           = set(working_text)


        while len(vocab) < vocab_size:

            #Find top pair
            pairs       = {} 
            i           = 0 

            #ATTEMPT 1 
            while i+1 < len(working_text):
                next_pair       = f"{working_text[i]}{working_text[i+1]}"
                next_pair       = "".join(working_text[i:i+2])

                if next_pair in pairs:
                    pairs[next_pair] += 1 
                else:
                    pairs[next_pair] =1 

                i += 1

            #END ATTEMPT
            top_pair    = max(pairs.items(),key= lambda x: x[1])

            #refactor working_text
            vocab.add(top_pair[0])
            print(f"ran one: {top_pair} - size={len(vocab)}")
            print(working_text[:25])

            new_working_text    = [] 
            working_text        = list(reversed(working_text))
            last_addition       = working_text.pop()
            chunk               = last_addition
            while working_text:

                while chunk in vocab:
                    try:
                        last_addition   = working_text.pop()
                        chunk           += last_addition
                        breakout        = False
                    except IndexError as ie:
                        new_working_text.append(chunk)
                        breakout        = True
                        break
                if breakout:
                    break
                
                new_working_text.append(chunk[:-len(last_addition)])
                chunk           = last_addition


            
            working_text        = new_working_text

        self.encoder    = {}
        self.decoder    = {}

        for i,word in enumerate(vocab):
            self.encoder[word]  = i
            self.decoder[i]     = word 
        return



    #encode one string of text  
    def encode(self,text:str,input_size:int,pad=True):
        for item in set(text):
            if not item in self.encoder:
                input(f"{item} -not found")
        #Prep items to convert to tokens
        text            = list(reversed(list(text)))
        tokens          = [] 

        last_addition   = text.pop()
        chunk           = last_addition

        while text:

            while chunk in self.encoder:
                try:
                    last_addition   = text.pop()
                    chunk           += last_addition
                    breakout        = False 
                except IndexError:
                    tokens.append(self.encoder[chunk])
                    breakout        = True 
                    break
                
            if breakout:
                break
            tokens.append(self.encoder[chunk[:-len(last_addition)]])
            chunk                   = last_addition
            if not text:
                tokens.append(self.encoder[chunk])
                break
        
        masks           = [1 for _ in tokens] + [0 for _ in range(input_size-len(tokens) if pad else 0)]
        #tokens          += [0 for _ in range(input_size-len(tokens) if pad else 0)]
        return tokens,masks


    def encode_batch(self,batches:list[str],input_size:int):
        token_masks_pairs   = [self.encode(batch,input_size) for batch in batches]
        tokens,masks    = ([],[])
        for pair in token_masks_pairs:
            tokens.append(pair[0])
            masks.append(pair[1])

        return tokens,masks 


    def decode(self,tokens:str):
        return "".join([self.decoder[tok] for tok in tokens])


    def save(self,path:str):

        if not os.path.exists(path):
            os.mkdir(path)

        tokenizer       = {"encoder":self.encoder,"decoder":self.decoder,"eos_token":self.eos_token,"pad_token":self.pad_token}

        with open(os.path.join(path,"tokenizer.json"),'w') as write_file:
            write_file.write(json.dumps(tokenizer))
        return


    def load(self,path:str):

        with open(os.path.join(path,"tokenizer.json"),'r') as read_file:
            tokenizer_json  = json.loads(read_file.read())
        
        self.encoder        = {k:int(v) for k,v in tokenizer_json['encoder'].items()}
        self.decoder        = {int(k):v for k,v in tokenizer_json['decoder'].items()}
        self.eos_token      = tokenizer_json['eos_token']
        self.pad_token      = tokenizer_json['pad_token']

        return
