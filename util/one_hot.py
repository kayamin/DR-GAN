#!/usr/bin/env python
# encoding: utf-8

import torch

def one_hot(label,depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,label)
