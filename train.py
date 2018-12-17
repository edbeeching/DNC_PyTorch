#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:20:29 2018

@author: edward
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torch import nn, Tensor
import torch.nn.functional as F


from differential_neural_computer import DNC, Params
from create_ntm_data import train_generator

params = Params()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = train_generator(num_bits=params.input_size-1, sequence_length=params.seq_length, batch_size=params.batch_size)

dnc = DNC(params).to(device)


loss_fn = nn.BCELoss()
optimizer= torch.optim.Adam(dnc.parameters(), lr=1e-3)
print('Printing layer names')
for name, param in dnc.named_parameters():
    if param.requires_grad:
        print(name)
        
for epoch in range(1000):
    permute = np.random.permutation(len(data))
    data = data[permute]
    losses = []
    for i in range(len(data) // params.batch_size - params.batch_size):
        
        batch = Tensor(data[i*params.batch_size: (i+1) * params.batch_size]).to(device)
        state = dnc.reset()
        for i in range(params.seq_length):
            piece = batch[:,i,:]
            out, state = dnc(piece, state)
        
        output = []
        piece = torch.zeros(params.batch_size, params.input_size).to(device)
        for i in range(params.seq_length):
            out, state = dnc(piece, state)            
            output.append(out)
        
        loss = loss_fn(F.sigmoid(torch.stack(output)), batch[:,:, :-1].transpose(0,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        losses.append(loss.item())
    
    pred = F.sigmoid(torch.stack(output))[0].detach().cpu().numpy()
    target = batch[:,:, :-1].transpose(0,1)[0].cpu().numpy()
    

    
    
    print(datetime.now(), epoch, sum(losses)/len(losses))

    plt.subplot(1,2,1)
    plt.imshow(target)
    plt.subplot(1,2,2)
    plt.imshow(pred)
    plt.savefig('examples/result_time{}_epoch{:003}.jpg'.format(datetime.now(), epoch))
    plt.close()

filename = 'models/model{}.pth.tar'.format(datetime.now())
with open(filename,'wb') as f:
    torch.save(dnc,f)