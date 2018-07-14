# Code reffered and adapted from ASSignment 1 Assignment 3 and https://github.com/samiroid/usr2vec
#and https://github.com/samiroid/CUE-CNN/tree/master/code
# coding: utf-8

# In[36]:


import argparse
import pickle
from collections import defaultdict
from collections import Counter
import numpy as np
import os
import time
import random
import tensorflow as tf
import parameters
import re
from nltk.tokenize import TweetTokenizer
from nltk import tokenize
np.set_printoptions(threshold=np.nan)
MIN_DOC_LEN=4
tokenizer = TweetTokenizer(preserve_case=True)
regex = re.compile(r'[\.\]\%\[\'",\?\*!\}\{<>\^-]')


# Create Negative Samples that are user specific and then using negative samples and the previous user word ids create user specific train data. create_user_train function is similar to the batch process of skigram model of word2vec.

# In[37]:


embed = "word2vec_nce.model"  #100000 * 5 lines from bz2 + user_text
train_file = 'clean_data.txt'  #sentences of 13746 users
output_pkl = "train_embeddings.pkl"
vocabulary_size = 100000
min_word_freq = 2
neg_samples = 10
output = "user_train_data.pkl"


# In[38]:


def load_embeddings(filename, wordDict, embedding_size):
    dictionary, steps, word_embeds = pickle.load(open(filename, 'rb'))
    print("Loading Embeddings.....")
    key = list(wordDict.keys())[0]
    embedding_array = np.zeros((len(wordDict), embedding_size))
    knownWords = list(wordDict.keys())
    unknownWords = []
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1

        w = knownWords[i]
        if w in dictionary:
            index = dictionary[w]
        elif w.lower() in dictionary:
            index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            unknownWords.append(w)
            embedding_array[i] = np.random.rand(embedding_size) * 0.02 - 0.01
    return embedding_array, dictionary, unknownWords


# In[39]:


def get_neg_samples(user_dict, wc):
    neg_dict = defaultdict(list)
    for user, message in user_dict.items():
        user_wrd = set(message)
        word_corpus = set(wc.keys())
        diff = word_corpus - user_wrd
        neg_dict[user] = random.sample(diff, parameters.sent_idx)
    return neg_dict


# In[40]:


# taken from negative_samples.py of samiroid
def multinomial_samples(unigram_distribution, exclude=[], n_samples=1):
    samples = []
    while len(samples) != n_samples:
        wrd_idx = np.argmax(np.random.multinomial(1, unigram_distribution))
        if wrd_idx not in exclude:
            samples.append(wrd_idx)
    return samples


# In[41]:


def create_embeds():
    t0 = time.time()
    print("Started create_embeds: ", t0)
    word_counter = Counter()
    n_docs = 0
    embedding_size = 128
    user_dict = defaultdict(list)
    c = 0
    with open(train_file,"r") as fid:
        for line in fid:
            if len(line) > 1:
                message = line.split(" ")
                content = tokenizer.tokenize(" ".join(message[2:]))
                content = [word for word in content if not regex.match(word)]

                user_dict[message[0]].extend(content)
                word_counter.update(content)
                n_docs += 1
    #keep only words that occur at least min_word_freq times
    wc = {w:c for w,c in word_counter.items() if c > min_word_freq}
    tw = sorted(wc.items(), key=lambda x:x[1], reverse=True)
    top_words = {w[0]:i for i,w in enumerate(tw)}

    embed_matrix, dictionary, unknownWords = load_embeddings(embed, top_words, embedding_size)
    wrd2idx = {w:i for i,w in enumerate(top_words.keys())}

    #finding unigram probability
    unigram_cnt = [c for w, c in top_words.items()]
    total = sum(unigram_cnt)
    unigram_prob = [c*1.0/total for c in unigram_cnt]

    #generate the embedding matrix
    print("embed_matrix shape",embed_matrix.shape)
    emb_size = embed_matrix.shape[1]
    print(emb_size)
    E = np.zeros((len(wrd2idx),int(emb_size)))
    for wrd,idx in wrd2idx.items():
        E[:] = embed_matrix[top_words[wrd],:]
    print("E shape",E.shape)
    pickle.dump([E, unigram_prob, wrd2idx, word_counter, len(user_dict.keys())], open(output_pkl, 'wb'))
    print("Finished create_embeds: ", time.time() - t0)

    return user_dict, wc, wrd2idx, n_docs, unigram_prob



# In[42]:


def create_user_train(user_dict, wc, wrd2idx, n_docs, unigram_prob):
    print("Started user train")
    start = time.time()
    prev_user, prev_user_data, prev_ctxscores, prev_neg_samples  = None, [], [], []
    full_train = []
    rng = np.random.RandomState(seed)
    c = 0
    with open(train_file, "r") as fid:
        for j, line in enumerate(fid):
            if len(line) <= 1:
                continue
            message = line.split(" ")
            user_id = message[0]
            content = tokenizer.tokenize(" ".join(message[2:]))
            content = [word for word in content if not regex.match(word)]

            negative_samples = multinomial_samples(unigram_prob, [], 10)
            #convert to indices
            msg_idx = [wrd2idx[w] for w in content if w in wrd2idx]
            if len(msg_idx) == 0:
                print("content: ", content)
                msg_idx = [0]

            if prev_user == None:  # first user
                prev_user = user_id
            elif user_id != prev_user or j == n_docs - 1: # this user_id is seen for the first time

                assert len(prev_user_data) == len(prev_neg_samples)
                shuf_idx = np.arange(len(prev_user_data))
                rng.shuffle(shuf_idx)

                # fill these lists with the same data in a different order
                prev_user_data = [prev_user_data[i] for i in shuf_idx]
                prev_neg_samples = [prev_neg_samples[i] for i in shuf_idx]

                split = int(len(prev_user_data)*.9)
                train = prev_user_data[:split]
                test  = prev_user_data[split:]
                neg_samples = prev_neg_samples[:split]

                # each training instance consists of:
                # [user_name, train docs, test docs, negative samples]
                full_train.append([prev_user, train, test, neg_samples])

                prev_user_data = []
                prev_neg_samples = []

            prev_user = user_id
            prev_user_data.append(msg_idx)
            prev_neg_samples.append(negative_samples)

        full_train.append([prev_user, train, test, neg_samples])
        print('prev user: ', prev_user)
        print('prev user data: ', train)
        print('prev neg samples: ', prev_neg_samples)

    print("Finished user_train", time.time() - start)
    pickle.dump(full_train, open(output, 'wb'))


# In[43]:


user_dict, wc, wrd2idx, n_docs, unigram_prob = create_embeds()

create_user_train(user_dict, wc, wrd2idx, n_docs, unigram_prob)
