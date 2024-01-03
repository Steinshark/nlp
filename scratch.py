from transformers import AutoTokenizer,GPT2Config,GPT2Tokenizer,GPT2LMHeadModel,Trainer,TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
import torch 
from torch.utils.data import Dataset
import utils 
from model import GPTSteinsharkDataSet,GPTSteinsharkTokenizer\

__DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#SETTINGS 
input_size  = 64
bs          = 4
cutoff      = None
epochs      = 8
lr          = 2e-4
save_ckpt   = 256 
vocab_size  = 512

#BUILD MODEL 
config      = GPT2Config(n_positions=input_size,n_head=8,n_layer=4,n_embed=1024,vocab_size=vocab_size) 
model       = GPT2LMHeadModel(config=config).to(__DEVICE)
tokenizer   = AutoTokenizer.from_pretrained('gpt2')
tokenizer   = tokenizer.train_new_from_iterator(GPTSteinsharkDataSet("corpus.txt",None,input_size,"<|ENDOFTEXT|>",None,is_dir=False).get_iter(max_i=2_000_000),vocab_size=vocab_size)
trainset    = GPTSteinsharkDataSet("corpus.txt",tokenizer=tokenizer,n_positions=input_size,eot_token="<|ENDOFTEXT|>",pad_token="<|PAD|>",is_dir=False)
tokenizer.pad_token     = "|PAD|"


#BUILD TRAINER
arguments           = TrainingArguments("C:/users/evere/temp",per_device_train_batch_size=bs,per_device_eval_batch_size=bs,num_train_epochs=epochs,evaluation_strategy="epoch",learning_rate=lr,save_steps=save_ckpt)
#trainset,testset    = utils.load_data("C:/code/nlp/data",input_size,tokenizer,cutoff=cutoff,eval_split=.01)
testset             = GPTSteinsharkDataSet("corpus.txt",tokenizer=tokenizer,n_positions=input_size,eot_token="<|ENDOFTEXT|>",pad_token="<|PAD|>",is_dir=False)
trainset.size       = 32
testset.size        = 32
trainer     = Trainer(model,args=arguments,train_dataset=trainset,eval_dataset=testset,tokenizer=tokenizer,data_collator=DataCollatorWithPadding(tokenizer=tokenizer,padding=True))
#print(f"trainset:\t{len(trainset)}\n\t{tokenizer.decode(trainset.__getitem__(0))[:128]}...")
#print(f"testset:\t{len(testset)}\n\t{tokenizer.decode(testset.__getitem__(0))[:128]}...")
print("\n\nBeginning Train")
trainer.train()


#BUILD TOKENIZER 


model_in    = tokenizer("I have a cat. My cat is:",return_tensors="pt")
model_in    = {key:model_in[key].to(__DEVICE) for key in model_in}
outputs     = model.generate(**model_in,max_length=input_size)


text        = tokenizer.decode(outputs[0])
print(f"output: {text}")
