from transformer import LMSteinshark
from tok import load_tokenizer
import json 
import torch 
import random 
from matplotlib import pyplot as plt 
from datasets import load_dataset
from huggingface_hub import login

#Get tokenizer 
tokenizer   = load_tokenizer("C:/gitrepos/cloudGPT/tokenizer")
model       = LMSteinshark.from_loadpoint("D:/nlp/models/production")
model.name  = "sorter"
for p in model.parameters():
    p.requires_grad_(False)

model.class_head = torch.nn.Sequential(torch.nn.Linear(2048,1)).cuda()

model       = model.bfloat16()

model_out   = model.generate("Hi guys, welcome back to the blog! Today we're going to talk about computers and building gaming PCs!",tokenizer=tokenizer)
#print(model_out)
model_in    = torch.tensor(tokenizer.encode(model_out).ids).unsqueeze(0).cuda()
#print(model.train_class_fwd(model_in)[0])

#Get data 
data    = json.loads(open("C:/gitrepos/nlp/traindata.json",'r').read())
pos     = data['positive']
neg     = data['negative']

dataset = [] 
pp     = 0 
nn     = 0
for data in pos:
    pp += 1
    dataset.append((torch.tensor(tokenizer.encode(data).ids),1))
for data in neg:
    nn += 1
    dataset.append((torch.tensor(tokenizer.encode(data).ids),-1))


print(f'Through with {model_in.shape} ({pp},{nn})')
bs = 1
optimizer   = torch.optim.AdamW(params=model.parameters(),lr=.0001)
c_loss      = []
avg_loss    = 0 
loss_fn     = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pp/nn]))         
for _ in range(int(2*len(dataset)/bs)):
    batch   = random.sample(dataset,k=bs)[0]

    x       = batch[0].unsqueeze(0)
    x       = x.long().cuda()
    y       = torch.tensor(batch[1]).unsqueeze(0).bfloat16().cuda().unsqueeze(0)


    y_      = model.train_class_fwd(x)

    loss    = torch.nn.functional.binary_cross_entropy_with_logits(y_,y)
    loss.backward()
    optimizer.step()
    #print(loss)
    avg_loss += float(loss.detach().cpu())

    if _ % 20 == 0:
        c_loss.append(avg_loss/20)
        avg_loss    = 0 

    # if _ % 500 == 0:
    #     print(f"saving on {_}")
    #     model.save('D:/nlp/models/')
    #     plt.plot(c_loss)
    #     plt.show()


model.set_generate_mode()
login(token=open('key.secret','r').read())

ds          = load_dataset("HuggingFaceFW/fineweb",data_dir="data/CC-MAIN-2024-38",streaming=True,split="train")

for sample in iter(ds):
    judged  = False 
    text    =   sample['text']
    inp     = torch.tensor(tokenizer.encode(text).ids).unsqueeze(0).cuda()

    score   = model.train_class_fwd(inp)

    print(f"text:\n{text}\n\nscore: {score.detach().cpu()}")
    input()