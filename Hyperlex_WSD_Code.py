#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import nltk
import re
from collections import Counter
from datetime import datetime
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
import sys


# In[2]:


def get_corpus_1st_2nd_neighbours(filename):
    file1= open(filename,"r")
    corpus = file1.read()
    corpus_in_list = re.findall(r"(?<=\(')[\w\s\.-]+(?=')", corpus)
    corpustext  = ""
    for c in corpus_in_list:
        corpustext = corpustext + " " + (c)
        
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    corpuslines = nltk.sent_tokenize(corpustext)
    collocation_list = []
    corpus_lemma = []
    corpus_pos = []

    for i in range(len(corpuslines)):
        sent_tokens = nltk.word_tokenize(corpuslines[i])
        lemmatized_sent,sent_postag = get_lemmatized_sentence(sent_tokens,tag_map)
        corpus_lemma.extend(lemmatized_sent)
        corpus_pos.extend(sent_postag)
    return corpus_lemma,corpus_pos


# In[3]:


def get_lemmatized_sentence(tokens,tag_map):
    lmtzr = WordNetLemmatizer()    
    lemmatized_sent =  []
    sent_tag = []
    for token, tag in nltk.pos_tag(tokens):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemmatized_sent.append(lemma)
        sent_tag.append(tag)
    return lemmatized_sent,sent_tag

def get_collocation_first_neighbour(tokens,tags,targetword):
    collocation_list = []
    collocation_tag_list = []
    #find direct neighbours
    for i in range(len(tokens)):
        start = max((i-3),0)
        end = min(i+4,len(tokens))
        if tokens[i] == targetword:
            collocation_list.extend(tokens[start:end])
            collocation_tag_list.extend(tags[start:end])
    collocation_list = remove_tag_stopwords(collocation_list,collocation_tag_list)
    
    return collocation_list

def remove_tag_stopwords(collocation_list,collocation_tag_list):
    tags_to_process = ['NNS','NN','JJ']
    words_to_process = [collocation_list[i] for i in range(len(collocation_tag_list)) if collocation_tag_list[i] in tags_to_process] 
    return words_to_process
    


# In[4]:


def get_word_Counter(word_list):
    word_Counter = Counter()
    for i in word_list:
        word_Counter[i] += 1
    return word_Counter


def get_all_collocation_for_target(corpus_lemma,corpus_pos,target):
       
    hyperlex_corp = defaultdict(Counter)
    col_1st_list = get_collocation_first_neighbour(corpus_lemma,corpus_pos,target)
    for w in col_1st_list:
        if w != target:
            hyperlex_corp[target][w] += 1
            hyperlex_corp[w][target] += 1
            
    col_2nd_list = []
    direct_neighbours = np.where(np.asarray(col_1st_list) != target)[0]

    for j in direct_neighbours:
        node = col_1st_list[j]
        col_2nd_list = get_collocation_first_neighbour(corpus_lemma,corpus_pos,node)
        
        for w in col_2nd_list:
            if w != node:
                hyperlex_corp[node][w] += 1
                hyperlex_corp[w][node] += 1
    #collocation_all = col_1st_list+col_2nd_list
    
    return hyperlex_corp

def get_top_senses(hyperlex_corp,target):
    max_key = max(hyperlex_corp[target], key=hyperlex_corp[target].get)
    top3_sense = {}
    top3_sense_list = []
    for key in sorted(hyperlex_corp[max_key], key=hyperlex_corp[max_key].get, reverse=True)[:3]:
        top3_sense.update({key: hyperlex_corp[max_key][key]})
    top3_sense_list = list(top3_sense.keys())
    top3_sense_list.insert(0, max_key)
    if top3_sense_list.count(target) > 0:
        top3_sense_list.remove(target)
    else:
        top3_sense_list.pop()
    return top3_sense_list



def delete_sense_words(hyperlex_corp,top_senses, targetword):
    sense_connected_nodes = list(hyperlex_corp[top_senses[0]].keys())
    if sense_connected_nodes.count(targetword) > 0:
        sense_connected_nodes.remove(targetword)
    sense_connected_nodes.append(top_senses[0])
    for s in sense_connected_nodes:
        if len(list(hyperlex_corp[s].keys()))>0:
            del hyperlex_corp[s]
        del hyperlex_corp[targetword][s]
    return hyperlex_corp

def get_topN_senses_for_corpus(cor_lemma,cor_pos,hyperlex_corp,targetword):
    all_senses_list = []
    i=0
    while len(list(hyperlex_corp[targetword].keys())) > 0:
        i = i+1
        sense_i = get_top_senses(hyperlex_corp,targetword)
        hyperlex_corp = delete_sense_words(hyperlex_corp, sense_i, targetword)
        #print(sense_i)
        sense_i_list = []
        sense_i_tuple = ()
        for s in range(len(sense_i)):
            sense_word= "<sense"+str(i)+"_"+sense_i[s]+">"
            sense_i_list.append(sense_word)
        sense_i_tuple = tuple(sense_i_list)
        all_senses_list.append(sense_i_tuple)
    print(tuple(all_senses_list))
        
        


# In[8]:


def main():
    arguments = sys.argv[1:]
    inputfilename = arguments[0]
    file1= open(inputfilename,"r")
    target = file1.read()
    cor_lemma,cor_pos = get_corpus_1st_2nd_neighbours('corpus.txt')
    hyperlex_corp = get_all_collocation_for_target(cor_lemma,cor_pos,target)
    get_topN_senses_for_corpus(cor_lemma,cor_pos,hyperlex_corp,target)
    

if __name__ == "__main__":
    main()

