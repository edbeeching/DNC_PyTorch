#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:05:15 2018

@author: edward
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F



def one_plus(x):
    return 1 + (1+x.exp()).log()

def index_slice(x, sizes):
    tot = 0
    indices = []
    for s in sizes:
        indices.append(s+tot)
        tot += s
        
        
    h,w = x.size()
    outputs = []
    cur = 0
    for ind in indices:
        assert ind <= w
        cur_slice = x[:, cur:ind]
        outputs.append(cur_slice)
    
        cur = ind
    return outputs
    
    
    

class DNC(nn.Module):
    def __init__(self, params):
        super(DNC, self).__init__()

        self.params = params
        self.controller = nn.LSTM(params.c_in_size, params.c_out_size)
        self.access = Access(params)
        self.linear = nn.Linear(params.l_in_size, params.l_out_size)

    def forward(self, x, state):
        read = state['read']
        c_state = state['c_state']
        memory = state['memory']
        a_state = state['a_state']
        # get the controller output
        control_out, c_state = self.controller(torch.cat([x, read], 0), c_state)

        # split the controller output into interface and control vectors

        read, a_state, memory  = self.access(control_out, a_state, memory)
        output = self.linear(torch.cat([control_out, read], 0))

        state = {'read': read,
                'c_state': c_state,
                'a_state': a_state,
                'memory': memory}

        return output, memory

    def reset_memory(self):
        pass

class Access(nn.Module):

    def __init__(self, params):
        super(Access, self).__init__()
        self.params = params
        self.interface_linear = nn.Linear(params.c_out_size, params.interface_size)
        self.read_heads = [ReadHead(params) for _ in range(params.num_read_heads)]
        self.write_head = ReadHead(params)
        
        self.interface_indices = [params.mem_size]*params.num_read_heads + [1]*params.num_read_heads \
                                  + [params.mem_size] + [1] \
                                  + [params.mem_size]*2 \
                                  + [params.mem_size] \
                                  + [1] + [1] + [params.mem_size]*params.num_read_heads
             
                                  
    def split(self, i_vec):
        i_vec_split = index_slice(i_vec, self.interface_indices)
        cur = 0
        r_keys = []
        r_betas = []
        # w_key = []
        # w_beta = []
        # e_vector = []
        # w_vector = []
        fgates  = []
        # a_gate = []
        # w_gate = []
        r_modes = []
        
        for i in range(self.params.num_read_heads):
            r_keys.append(i_vec_split[cur])
            cur += 1
            
        for i in range(self.params.num_read_heads):
            r_betas.append(i_vec_split[cur])
            cur += 1
        w_key = i_vec_split[cur]
        cur += 1
        w_beta = i_vec_split[cur]
        cur += 1
        e_vector = i_vec_split[cur]
        cur += 1
        w_vector = i_vec_split[cur]
        cur += 1
        
        for i in range(self.params.num_read_heads):
            fgates.append(i_vec_split[cur])
            cur += 1       
        
        a_gate = i_vec_split[cur]
        cur += 1
        w_gate = i_vec_split[cur]
        cur += 1       
        
        for i in range(self.params.num_read_heads):
            r_modes.append(i_vec_split[cur])
            cur += 1        
        
        return r_keys, r_betas, w_key, w_beta, e_vector, w_vector, fgates , a_gate, w_gate, r_modes
        

    def forward(self, x, memory):
        interface_vector = self.interface_linear(x)
        
        split = self.split(interface_vector)
        r_keys, r_betas, w_key, w_beta, e_vector, w_vector, fgates , a_gate, w_gate, r_modes = split
        # read the memory
        reads = []
        for read_head, r_key, r_beta, r_mode in zip(self.read_heads, r_keys, r_betas, r_modes):
            r_beta = one_plus(r_beta)
            r_mode = F.softmax(r_mode,1)
            
            read = read_head(r_key, r_beta, r_mode, memory)
            reads.append(read)
            
        # write to the momory
        memory = self.write_head(w_key, w_beta, e_vector, w_vector, memory)
        
        return torch.cat(reads, 1), memory

class HeadFuncs:
    @staticmethod
    def query(key, memory):
        if len(key.size()) < len(memory.size()):
            print('resizing key')
            h,w = key.size()
            key = key.view(h,1,w)
        m = memory / (memory.norm(2, dim=2, keepdim=True) + 1E-8 )
        k = key / (key.norm(2, dim=1, keepdim=True) + 1E-8)
        
        weights = (m*k).sum(-1)
        
        return F.softmax(weights, dim=1)
        
        
        
        

        
class ReadHead(nn.Module):
    def __init__(self, params):
        super(ReadHead, self).__init__()
        pass
    
    def forward(self, r_key, r_beta, r_mode, memory):
        
        
        
        pass

class WriteHead(nn.Module):
    def __init__(self, params):
        super(ReadHead, self).__init__()
        pass
    def forward(self, x, memory):
        
        return x, memory




if __name__ == "__main__":
    

    key = torch.randn(2,5)
    memory = torch.randn(2,4,5)
    
    weights = HeadFuncs.query(key, memory)



