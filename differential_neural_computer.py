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
        link_matrix = state['link_matrix']
        # get the controller output
        control_out, c_state = self.controller(torch.cat([x, read], 0), c_state)

        # split the controller output into interface and control vectors
        read, memory, a_state, link_matrix  = self.access(control_out, memory, a_state, link_matrix)
        output = self.linear(torch.cat([control_out, read], 0))

        state = {'read': read,
                'c_state': c_state,
                'a_state': a_state,
                'memory': memory,
                'link_matrix': link_matrix}

        return output, state

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
        r_weights = []
        for read_head, r_key, r_beta, r_mode in zip(self.read_heads, r_keys, r_betas, r_modes):
            r_mode = F.softmax(r_mode, 1)
            read, r_weight = read_head(r_key, r_beta, r_mode, memory)
            reads.append(read)
            r_weights.append(r_weight)

        return torch.cat(reads, 1), memory, (torch.cat(r_weights,1), w_weights, usage, precedence)

class HeadFuncs:
    @staticmethod
    def query(key, beta, memory):
        if len(key.size()) < len(memory.size()):
            #print('resizing key')
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
        pass
    
    def forward(self, r_key, r_beta, r_mode, memory):
        weights = HeadFuncs.query(r_key, r_beta, memory)
        return (memory*weights.unsqueeze(2)).sum(1), weights

class WriteHead(nn.Module):
    def __init__(self, params):
        super(ReadHead, self).__init__()
        pass
    
    @staticmethod
    def update_link(link, weights, precedence):
        
        b,h,w = link.size()            
        assert h == w
        assert weights.size() == precedence.size()
        w_i = weights.unsqueeze(2).repeat(1,1,w)
        w_j = weights.unsqueeze(1).repeat(1,h,1)
        p_j = prec.unsqueeze(1).repeat(1,h,1)
        
        link = (1 - w_i - w_j)*link + w_i*p_j
        
        mask = 1 -torch.eye(h).unsqueeze(0).repeat(b,1,1)
        link = link*mask        
        
        return link

    
    def forward(self, w_key, w_beta, e_vector, w_vector, a_gate, w_gate, allocations, memory, link_matrix, precedence):
        c_weights = HeadFuncs.query(w_key, w_beta, memory)
        assert (c_weights.size() == allocations.size() == precedence.size())
        assert (w_key.size() == e_vector.size() == w_vector.size())
        w_gate = F.sigmoid(w_gate)
        a_gate = F.sigmoid(a_gate)
        
        w_weights = w_gate*(a_gate*allocations + (1-a_gate)*c_weights)
        
        
        
        
        
        
        
        e_vector = F.sigmoid(e_vector)
        
        

        link_matrix = self.update_link(link_matrix, w_weights, precedence)
        precedence = (1 - w_weights.sum(1, keepdum=True)) * precedence + w_weights
        
        return w_weights, memory, link_matrix, precedence


def link1(link, weights, prec):
    
    b,h,w = link.size()
    
    link_1 = torch.zeros_like(link)
    for i in range(h):
        for j in range(w):
            if i==j: continue
            link_1[:, i,j] = (1 - weights[:,i] - weights[:,j])*link[:,i,j] + weights[:, i]*prec[:, j]

    return link_1
            
def link2(link, weights, prec):
    
    b,h,w = link.size()            
    assert h == w
    w_i = weights.unsqueeze(2).repeat(1,1,w)
    w_j = weights.unsqueeze(1).repeat(1,h,1)
    p_j = prec.unsqueeze(1).repeat(1,h,1)
    
    link2 = (1 - w_i - w_j)*link + w_i*p_j
    
    mask = 1 -torch.eye(h).unsqueeze(0).repeat(b,1,1)
    link2 = link2*mask
    
    return link2
    

if __name__ == "__main__":
    
    size = 4

    weights = F.softmax(torch.rand(2, size),1)
    prec = F.softmax(torch.rand(2, size),1)

    link = torch.rand(2,size,size)
    
    link_1 = link1(link, weights, prec)
    print(link_1)
    link_2 = link2(link, weights, prec)
    print(link_2)
    link_3 = WriteHead.update_link(link, weights, prec)
    print(link_3)
    print(link_1 == link_2)
    print(link_1 == link_3)
    
