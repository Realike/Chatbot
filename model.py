from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from vocabulary import PAD_token, SOS_token, EOS_token, UNK_token, Voc

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import pickle

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


#
# # load voc and pairs
def loadDataset():
    with open('voc.pkl', 'rb') as handle_voc:
        voc = pickle.load(handle_voc)

    with open('pairs.pkl', 'rb') as handle_pairs:
        pairs = pickle.load(handle_pairs)

    return voc, pairs


voc, pairs = loadDataset()


#
# # add EOS_token
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


#
# 通过zip_longest()，zeroPadding时矩阵转置
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


#
# mask矩阵
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)

    return m


#
# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]  # word2index batch
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)  # list to longTensor,shape(max_length, batch_size)
    # padVar = torch.tensor(padList, dtype=torch.long)

    return padVar, lengths


#
# # Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    padVar = torch.LongTensor(padList)  # not torch.longTensor
    # padVar = torch.tensor(padList, dtype=torch.long)

    return padVar, mask, max_target_len


#
# # Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)

    return inp, lengths, output, mask, max_target_len


#
# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print('input_variable:', input_variable)
print('length:', lengths)
print('target_variable:', target_variable)
print('mask:', mask)
print('max_target_len:', max_target_len)
