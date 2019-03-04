from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data', corpus_name)

#
# # testing 10 lines corpus
def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# printLines(os.path.join(corpus, 'movie_lines.txt'))

# #
# Splits each line of the file into a dictionary of fields
'''
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    split.(b'L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!\n')
'''
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj

    return lines

# #
# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj['utteranceIDs'])
            # Reassemble lines 每一句lines[lineId]初始化convObj，导入conversations
            convObj['lines'] = []
            for lineId in lineIds:
                convObj['lines'].append(lines[lineId])
            conversations.append(convObj)

    return conversations

# #
# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation['lines']) - 1):
            inputLine = conversation['lines'][i]['text'].strip()
            targetLine = conversation['lines'][i+1]['text'].strip()
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])

    return qa_pairs

# #
# create the file. We’ll call it formatted_movie_lines.txt.
datafile = os.path.join(corpus, 'formatted_movie_lines.txt')

delimiter = '\t'
# unescape the delimiter
delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
MOVIE_CONVERSATIONS_FIELDS = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']

# Load lines and process conversaions
print('\nProcessing corpus...')
lines = loadLines(os.path.join(corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
print('\nLoading conversations...')
conversations = loadConversations(os.path.join(corpus, 'movie_conversations.txt'), lines, MOVIE_CONVERSATIONS_FIELDS)

# write new csv files
print('\nWriting newly formatted file...')
with open(datafile, 'w', encoding='utf-8') as o:
    writer = csv.writer(o, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

# # Print a sample of lineIds
# print('\nSample lines from file:')
# printLines(datafile)

# Default word tokens ?UNK_token
PAD_token = 0   # padding token
SOS_token = 1   # start token
EOS_token = 2   # end token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.Trimmed =True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words),
            len(self.word2index), len(keep_words) / len(self.word2index)))


        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)
