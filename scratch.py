from transformers import GPT2Model,GPT2Config,GPT2Tokenizer,GPT2LMHeadModel,Trainer,TrainingArguments, DataCollatorForLanguageModeling
import torch 
from torch.utils.data import Dataset
import utils 



#SETTINGS 
input_size  = 512
bs          = 4
cutoff      = 128
epochs      = 4
lr          = 2e-5
save_ckpt   = 128 

#BUILD MODEL 
config      = GPT2Config(n_positions=input_size) 
model       = GPT2LMHeadModel(config=config)
tokenizer   = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token     = "|PAD|"


#BUILD TRAINER
tokenizer.model_max_length  = 1_000_000_000 #Avoid warning 
arguments           = TrainingArguments("C:/users/evere/temp",per_device_train_batch_size=bs,per_device_eval_batch_size=bs,num_train_epochs=epochs,evaluation_strategy="epoch",learning_rate=lr,save_steps=save_ckpt,resume_from_checkpoint="C:/users/evere/temp")
trainset,testset    = utils.load_data("C:/code/nlp/data",input_size,tokenizer,cutoff=cutoff,eval_split=.05)
trainset            = utils.GPTDataSet(trainset)
testset             = utils.GPTDataSet(testset)

trainer     = Trainer(model,args=arguments,train_dataset=trainset,eval_dataset=testset,tokenizer=tokenizer,data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False))
print(f"trainset:\t{len(trainset)}\n\t{tokenizer.decode(trainset[0]['input_ids'])[:128]}...")
print(f"testset:\t{len(testset)}\n\t{tokenizer.decode(testset[0]['input_ids'])[:128]}...")
print("\n\nBeginning Train")
tokenizer.model_max_length  = input_size
trainer.train()


#BUILD TOKENIZER 


model_in    = tokenizer("The model ",return_tensors="pt")
outputs     = model.generate(**model_in,max_length=256)


text        = tokenizer.decode(outputs[0])
print(f"output: {text}")
