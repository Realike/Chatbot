from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
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
import numpy as np
import sys

from vocabulary import PAD_token, SOS_token, EOS_token, Voc
from model import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, trainIters, evaluateInput

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

MAX_LENGTH = 15  # Maximun sentence length to consider

#
# # load voc and pairs
def loadDataset():
    with open(os.path.join('data', 'voc.pkl'), 'rb') as handle_voc:
        voc = pickle.load(handle_voc)

    with open(os.path.join('data', 'pairs.pkl'), 'rb') as handle_pairs:
        pairs = pickle.load(handle_pairs)

    return voc, pairs


voc, pairs = loadDataset()

save_dir = os.path.join('data', 'save')
corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data', corpus_name)

# 配置模型
model_name = 'cb_model'
# attn_model = 'dot'
attn_model = 'general'
# attn_model = 'concat'
hidden_size = 256
encoder_n_layers = 3
decoder_n_layers = 3
dropout = 0.1
batch_size = 64

# 配置超参数和优化器
clip = 5.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 100000
print_every = 100
save_every = 10000

# 从哪个checkpoint恢复，如果是None，那么从头开始训练。
loadFilename = None
checkpoint_iter = 4000

# 如果loadFilename不空，则从中加载模型
if loadFilename:
    # 如果训练和加载是一条机器，那么直接加载
    checkpoint = torch.load(loadFilename)
    # 否则比如checkpoint是在GPU上得到的，但是我们现在又用CPU来训练或者测试，那么注释掉下面的代码
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initializing word embedding
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)

# initializing encoder and decoder
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words,
                                decoder_n_layers, dropout)

if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

# use device
encoder = encoder.to(device)
decoder = decoder.to(device)

# 设置进入训练模式，从而开启dropout
encoder.train()
decoder.train()

# 初始化优化器
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

#
# # training
def train():
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                print_every, save_every, clip, corpus_name, loadFilename, hidden_size, teacher_forcing_ratio)

#
# # testing
def test():
    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)

    evaluateInput(encoder, decoder, searcher, voc)


# train() and test()
def main(argv):
    if argv[1] == 'train':
        train()
    elif argv[1] == 'test':
        test()


if __name__ == "__main__":
    main(sys.argv)
