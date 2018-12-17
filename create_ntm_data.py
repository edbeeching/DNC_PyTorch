#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:24:01 2018

@author: edward
"""
import numpy as np
import matplotlib.pyplot as plt


def gen_sequence(num_bits=7, sequence_length=20):
    sequence = np.random.binomial(1, 0.5, (sequence_length, num_bits+1))
    sequence[:,num_bits] = 1.0

    return sequence.astype(np.float32)


def train_generator(num_bits=7, sequence_length=20, batch_size=16, num_examples=1024):
    data = []
    
    for i in range(num_examples):
        seq = gen_sequence(num_bits=num_bits, sequence_length=sequence_length)
        data.append(seq)
    
    data = np.stack(data)
    return data


if __name__ == '__main__':
    

    train_generator()

