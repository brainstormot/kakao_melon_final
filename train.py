#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import json
import pandas as pd
import numpy as np
import pickle
import itertools
import operator
from tqdm import tqdm
from datetime import datetime
from arena_tool import remove_seen
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors


# In[ ]:


MODEL_PATH = './model/'
FILE_PATH = './data/raw/'
with open(os.path.join(FILE_PATH, 'train.json'), encoding="utf-8") as f:
    train = json.load(f)
with open(os.path.join(FILE_PATH, 'test.json'), encoding="utf-8") as f:
    test = json.load(f)
with open(os.path.join(FILE_PATH, 'val.json'), encoding="utf-8") as f:
    val = json.load(f)
with open(os.path.join(FILE_PATH, 'song_meta.json'), encoding="utf-8") as f:
    song_meta = json.load(f)

train_data  = pd.read_json(os.path.join(FILE_PATH,'train.json') , typ = 'frame')
test_data   = pd.read_json(os.path.join(FILE_PATH,'test.json') , typ = 'frame')
valid_data  = pd.read_json(os.path.join(FILE_PATH,'val.json'), typ = 'frame')

new_data = pd.concat([train_data,test_data,valid_data])

#%% prefare train data
plylist_title = new_data["plylst_title"].apply(lambda x: x.split(' ')).explode().value_counts()
relevant_title_word = set(plylist_title[plylist_title>1].index)

def make_data(data):
    result = []
    for q in tqdm(data):
        represent = []

        q['updt_data'] = pd.to_datetime(q['updt_date'])

        # plylst title processing
        title_word = []
        title = q['plylst_title'].split()
        for word in title:
            if word in relevant_title_word:
                title_word.append('f_' + str(word))

        tag   = q['tags']
        song  = ["s-"+str(song) for song in q['songs']]
        represent = song + tag + title_word

        q['repr'] = represent
        result.append(represent)
    return result

train_repr = make_data(data = train)
val_repr = make_data(data = val)
test_repr = make_data(data = test)
tot_repr = train_repr + val_repr + test_repr


# In[ ]:


# modeling w2v model# #

def hash(astring):
   return ord(astring[0])

min_count = 2
size = 200
window = 150
sg = 1

w2v_model = Word2Vec(tot_repr, 
                min_count= min_count,
                size = size, 
                window = window, 
                sg =sg,
                workers=8 ,
                hashfxn = hash 
            )

#%% save the w2v model
with open(os.path.join(MODEL_PATH,'0007ky.w2v'), 'wb') as f:
    pickle.dump(w2v_model, f)


# In[ ]:


# make p2v model
p2v_model = WordEmbeddingsKeyedVectors(size)
#%%
tot = train + test + val
song_dic = {}
tag_dic = {}
for q in tqdm(tot):
    song_dic[q['id']] = q['songs']
    tag_dic[q['id']] = q['tags']

#%%
ID = []   
vec = []
for q in tqdm(tot, leave= True , position= 0):
    tmp_vec = 0
    if len(q['repr'])>=1:
        for word in q['repr']:
            try: 
                tmp_vec += w2v_model.wv.get_vector(str(word))
            except KeyError:
                pass
    if type(tmp_vec)!=int:
        ID.append(str(q['id']))    
        vec.append(tmp_vec)
p2v_model.add(ID, vec)

with open(os.path.join(MODEL_PATH,'0007ky.p2v'), 'wb') as f:
    pickle.dump(p2v_model, f)

