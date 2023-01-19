# import
import os,sys
import time
from tqdm import tqdm
import random

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import SparseAdam
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim

import polars as pl
from merlin.loader.torch import Loader
from merlin.io import Dataset as mDataset



ROOT=os.path.dirname(os.getcwd())
sys.path.append(ROOT)
SEED=1024
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(SEED)
USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda') if USE_CUDA else torch.device('cpu')



# hyper parameter
NEG_N = 2
EMBEDDING_DIM = 32
BATCH_SIZE = 65536
LR = 0.1
EPOCHS=5
print('config setted')

# dataset
cardinality_aids =1855602
train_ds = mDataset('./data/train_pairs.parquet', engine="parquet")
train_dl_merlin = Loader(train_ds, BATCH_SIZE,drop_last=True)
valid_ds = mDataset('./data/valid_pairs.parquet')
valid_dl_merlin = Loader(valid_ds, BATCH_SIZE,drop_last=True)

# model
class Wrod2vec(nn.Module):
    def __init__(self, n_aids, n_factors):
        super().__init__()
        self.w1 = nn.Embedding(n_aids, n_factors, sparse=True)
        self.w2 = nn.Embedding(n_aids, n_factors, sparse=True)

    def forward(self,target,pos):
        hidden= self.w1(target).transpose(1,2) # b,1,d
        embed_pos= self.w2(pos) # b,1,d
        embed_neg=self.w2(torch.randint(0,cardinality_aids,[BATCH_SIZE,NEG_N],device=DEVICE)) # b,n,d
        return (-F.logsigmoid(torch.bmm(embed_pos, hidden).squeeze(2).sum(1)) - F.logsigmoid(
            torch.bmm(embed_neg, -hidden).squeeze(2).sum(1))).mean()
# eval
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

model = Wrod2vec(cardinality_aids + 1, EMBEDDING_DIM)
optimizer = SparseAdam(model.parameters(), lr=LR)


# train
if USE_CUDA:
    model.cuda()
for epoch in range(EPOCHS):
    for batch, _ in tqdm(train_dl_merlin):
        model.train()
        losses = AverageMeter('Loss', ':.4e')
        input,pos = batch['aid'].to(DEVICE), batch['aid_next'].to(DEVICE)
        loss =model(input,pos)
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        accuracy = AverageMeter('accuracy')
        for batch, _ in valid_dl_merlin:
            input, pos = batch['aid'].to(DEVICE), batch['aid_next'].to(DEVICE)
            hidden = model.w1(input).transpose(1,2)
            embed_pos = model.w2(pos)
            embed_neg = model.w2(pos[torch.randperm(pos.shape[0])])
            output_pos=torch.bmm(embed_pos,hidden).squeeze()
            output_neg=torch.bmm(embed_neg,hidden).squeeze()
            accuracy_batch = torch.cat([output_pos.sigmoid() > 0.5, output_neg.sigmoid() < 0.5]).float().mean()
            accuracy.update(accuracy_batch, input.shape[0])

    print(f'{epoch + 1:02d}: * TrainLoss {losses.avg:.3f}  * Accuracy {accuracy.avg:.3f}')
