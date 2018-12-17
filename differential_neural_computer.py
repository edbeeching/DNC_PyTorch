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
        self.controller = nn.LSTMCell(params.c_in_size, params.c_out_size)
        self.access = Access(params)
        self.linear = nn.Linear(params.l_in_size, params.l_out_size)
        
        nn.init.orthogonal_(self.controller.weight_hh)
        nn.init.orthogonal_(self.controller.weight_ih)
        nn.init.orthogonal_(self.linear.weight)
        self.linear.weight.data.fill_(0.0)
        
        

    def forward(self, x, state):
        read = state['read']
        c_state = state['c_state']
        memory = state['memory']
        a_state = state['a_state']
        link_matrix = state['link_matrix']
        # get the controller output
        h,c = c_state
        control_out, cell_state = self.controller(torch.cat([x, read], 1), c_state)

        c_state = ( control_out, cell_state)

        # split the controller output into interface and control vectors
        read, memory, a_state, link_matrix  = self.access(control_out, memory, a_state, link_matrix)
        
        output = self.linear(torch.cat([control_out, read], 1))
 
        
        state = {'read': read,
                'c_state': c_state,
                'a_state': a_state,
                'memory': memory,
                'link_matrix': link_matrix}

        return output, state

    def reset(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        read = torch.zeros(self.params.batch_size, 
                         self.params.num_read_heads * self.params.mem_size).to(device)
        
        hidden = torch.zeros(self.params.batch_size, self.params.c_out_size).to(device)
        cells = torch.zeros(self.params.batch_size, self.params.c_out_size) .to(device)
        c_state = (hidden,cells)
        memory = torch.zeros(self.params.batch_size, self.params.memory_n, self.params.mem_size).to(device)
        link_matrix = torch.zeros(self.params.batch_size, self.params.memory_n, self.params.memory_n).to(device)
        
        r_weights = torch.zeros(self.params.batch_size, self.params.num_read_heads, self.params.memory_n).to(device)
        w_weights = torch.zeros(self.params.batch_size, self.params.memory_n).to(device)
        usage = torch.zeros(self.params.batch_size, self.params.memory_n).to(device)
        precedence = torch.zeros(self.params.batch_size, self.params.memory_n).to(device)
        
        a_state = r_weights, w_weights, usage, precedence
        
        state = {'read': read,
                'c_state': c_state,
                'a_state': a_state,
                'memory': memory,
                'link_matrix': link_matrix}       
        
        return state
    
        
        
        

class Access(nn.Module):
    def __init__(self, params):
        super(Access, self).__init__()
        self.params = params
        self.read_heads = [ReadHead() for _ in range(params.num_read_heads)]
        self.write_head = WriteHead()
        
        self.interface_indices = [params.mem_size]*params.num_read_heads + [1]*params.num_read_heads \
                                  + [params.mem_size] + [1] \
                                  + [params.mem_size]*2 \
                                  + [1]*params.num_read_heads \
                                  + [1] + [1] + [3]*params.num_read_heads # 3 read modes
                                  
        interface_size = sum(self.interface_indices)
        self.interface_linear = nn.Linear(params.c_out_size, interface_size)
        
        nn.init.orthogonal_(self.interface_linear.weight)
        
        self.interface_linear.bias.data.fill_(0.0)
             
                                  
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
        
        return r_keys, r_betas, w_key, w_beta, e_vector, w_vector, fgates, a_gate, w_gate, r_modes
 
    @staticmethod
    def get_allocations(r_weights, w_weights, usage, f_gates):
        retention = torch.prod(1 - F.sigmoid(torch.cat(f_gates, 1)).unsqueeze(2)*r_weights, 1)
        usage = (usage + w_weights - usage * w_weights) * retention
        sorted_usage, inds  = usage.sort(1)
        usage_cumprod = torch.cumprod(sorted_usage, 1)
        before_scat = (1-sorted_usage)*usage_cumprod
        allocations = torch.zeros_like(usage).scatter_(1, inds, before_scat)        
        
        return allocations, usage

    def forward(self, x, memory, a_state, link_matrix):
        r_weights, w_weights, usage, precedence = a_state
        interface_vector = self.interface_linear(x)
        
        split = self.split(interface_vector)
        r_keys, r_betas, w_key, w_beta, e_vector, w_vector, f_gates, a_gate, w_gate, r_modes = split
        
        allocations, usage = self.get_allocations(r_weights, w_weights, usage, f_gates)
              
        # write to the momory
        w_weights, memory, link_matrix, precedence = self.write_head(w_key, w_beta, e_vector, w_vector, a_gate, w_gate, allocations, memory, link_matrix, precedence)    
    
        # read the memory
        reads = []
        r_weights_out = []
        for i in range(self.params.num_read_heads):
            read, r_weight = self.read_heads[i](r_keys[i], r_betas[i], r_modes[i], r_weights[:, i], memory, link_matrix)
            reads.append(read.squeeze(1))
            r_weights_out.append(r_weight)
        return torch.cat(reads, 1), memory, (torch.stack(r_weights_out, 1), w_weights, usage, precedence), link_matrix

class HeadFuncs:
    @staticmethod
    def query(key, beta, memory):
        if len(key.size()) < len(memory.size()):
            h,w = key.size()
            key = key.view(h,1,w)
            
        beta = one_plus(beta)
        # normalize the key and memory
        m = memory / (memory.norm(2, dim=2, keepdim=True) + 1E-8 )
        k = key / (key.norm(2, dim=1, keepdim=True) + 1E-8)
        
        weights = (m*k).sum(-1)
        
        return  F.softmax(weights*beta, dim=1)
   
class ReadHead(nn.Module):
    def __init__(self):
        super(ReadHead, self).__init__()
    
    def forward(self, r_key, r_beta, r_mode, r_weights, memory, link_matrix):
        r_mode = F.softmax(r_mode, 1)
        c_weights = HeadFuncs.query(r_key, r_beta, memory)
        f_weights = torch.bmm(link_matrix, r_weights.unsqueeze(2)).squeeze(2)
        b_weights = torch.bmm(link_matrix.transpose(1,2), r_weights.unsqueeze(2)).squeeze(2)
        # slice to retrain original dims
        weights = r_mode[:,0:1]*b_weights + r_mode[:,1:2]*c_weights + r_mode[:,2:3] * f_weights
        
        return torch.bmm(weights.unsqueeze(1), memory), weights

class WriteHead(nn.Module):
    def __init__(self):
        super(WriteHead, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def update_link(link, weights, precedence):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b,h,w = link.size()            
        assert h == w
        assert weights.size() == precedence.size()
        w_i = weights.unsqueeze(2).repeat(1,1,w)
        w_j = weights.unsqueeze(1).repeat(1,h,1)
        p_j = precedence.unsqueeze(1).repeat(1,h,1)
        
        link = (1 - w_i - w_j)*link + w_i*p_j
        
        mask = 1 -torch.eye(h).unsqueeze(0).repeat(b,1,1).to(device)
        link = link*mask        
        
        return link

    
    def forward(self, w_key, w_beta, e_vector, w_vector, a_gate, w_gate, allocations, memory, link_matrix, precedence):
        c_weights = HeadFuncs.query(w_key, w_beta, memory)
        
        b,h,w = memory.size()
        assert (c_weights.size() == (b,h))
        assert (w_vector.size() == (b,w))
        assert (w_key.size() == (b,w))
        
        assert (c_weights.size() == allocations.size() == precedence.size())
        assert (w_key.size() == e_vector.size() == w_vector.size())
        w_gate = F.sigmoid(w_gate)
        a_gate = F.sigmoid(a_gate)
        e_vector = F.sigmoid(e_vector)
        
        w_weights = w_gate*(a_gate*allocations + (1-a_gate)*c_weights)        
        link_matrix = self.update_link(link_matrix, w_weights, precedence)
        
        memory = memory*(torch.ones_like(memory) \
                - torch.bmm(w_weights.unsqueeze(2), e_vector.unsqueeze(1))) \
                + torch.bmm(w_weights.unsqueeze(2), w_vector.unsqueeze(1))
        
        precedence = (1 - w_weights.sum(1, keepdim=True)) * precedence + w_weights
        
        return w_weights, memory, link_matrix, precedence

    
class Params:
    def __init__(self):
        self.input_size = 8
        self.c_out_size = 64
        self.l_out_size = self.input_size-1
        self.mem_size = 16
        self.memory_n = 32    
        self.batch_size = 16
        self.num_read_heads = 3
        self.seq_length = 20
        
        self.l_in_size = self.c_out_size + self.num_read_heads * self.mem_size
        self.c_in_size = self.input_size + self.num_read_heads * self.mem_size        
    

if __name__ == "__main__":
       
    params = Params()

    dnc = DNC(params)

    read = torch.zeros(params.batch_size, 
                             params.num_read_heads*params.mem_size)
    
    hidden = torch.zeros(params.batch_size, params.c_out_size)
    cells = torch.zeros(params.batch_size, params.c_out_size) 
    c_state = (hidden,cells)
    memory = torch.zeros(params.batch_size, params.memory_n, params.mem_size)
    link_matrix = torch.zeros(params.batch_size, params.memory_n, params.memory_n)
    
    r_weights = torch.zeros(params.batch_size, params.num_read_heads, params.memory_n)
    w_weights = torch.zeros(params.batch_size, params.memory_n)
    usage = torch.zeros(params.batch_size, params.memory_n)
    precedence = torch.zeros(params.batch_size, params.memory_n)
    
    
    a_state = r_weights, w_weights, usage, precedence
    
    state = {'read': read,
            'c_state': c_state,
            'a_state': a_state,
            'memory': memory,
            'link_matrix': link_matrix}
    
    x = torch.randn(params.batch_size, params.input_size)
    
    
    for i in range(1000):
        x, state2 = dnc(x, state)
    