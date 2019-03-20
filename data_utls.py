from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from vocabulary import PAD_token, SOS_token, EOS_token, Voc

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
conversations = loadConversations(
    os.path.join(corpus, 'movie_conversations.txt'), lines, MOVIE_CONVERSATIONS_FIELDS)

# write new csv files
print('\nWriting newly formatted file...')
with open(datafile, 'w', encoding='utf-8') as o:
    writer = csv.writer(o, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

# # Print a sample of lineIds
# print('\nSample lines from file:')
# printLines(datafile)


MAX_LENGTH = 15  # Maximun sentence length to consider


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)   # 标点前增加空格
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)   # 字母和标点之外变成空格
    s = re.sub(r'\s+', r' ', s).strip()    # 由上可能导致多个空格，把多个空格变为一个空格并去首尾空格

    return s


# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print('Reading lines...')
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)

    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print('\nStart preparing training data...')
    voc, pairs = readVocs(datafile, corpus_name)
    print('Read {!s} sentence pairs'.format(len(pairs)))
    pairs = filterPairs(pairs)
    print('Trimmed to {!s} sentence pairs'.format(len(pairs)))
    print('Counting words...')
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print('Counted words:', voc.num_words)
    return voc, pairs


# Load/Assemble voc and qa_pairs
save_dir = os.path.join('data', 'save')
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print('\npairs:')
for pair in pairs[:10]:
    print(pair)
print('\n')

MIN_COUNT = 2   # Minimum word count threshold for trimming


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(
        len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

#
# store voc and pairs
print('Storing voc and pairs')
file_voc = open('data/voc.pkl', 'wb')
file_pairs = open('data/pairs.pkl', 'wb')
# file_voc = open(os.path.join('data', 'voc.pkl'), 'wb')
# file_pairs = open(os.path.join('data', 'pairs.pkl'), 'wb')
pickle.dump(voc, file_voc)
pickle.dump(pairs, file_pairs)
file_voc.close()
file_pairs.close()
print('Storing compeletely.')
