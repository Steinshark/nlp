import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
from torch.optim import Adam
import sys 
import os 
from transformer import LMSteinshark
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['USE_LIBUV'] = "0" 
def train(rank, world_size):
    dist.init_process_group("gloo", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Model, optimizer, criterion
    model = LMSteinshark(trig_embd=False).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Data loading
    dataset = FakeData(transform=ToTensor())
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for inputs, targets in dataloader:
            inputs      = torch.randint(0,32768-1,size=(16,512)).long().to(rank)
            targets     = torch.randint(0,32768-1,size=(16,512)).long().to(rank)
            #inputs, targets = inputs.to(rank).to(torch.long), targets.to(rank).to(torch.long)
            optimizer.zero_grad()

            outputs = model(inputs,targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)