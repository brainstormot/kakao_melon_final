#!/usr/bin/env python
# coding: utf-8

# # SONG ALS

# In[1]:


import pandas as pd
import numpy as np
import json
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares as ALS
from implicit.evaluation import  *
import os
from sklearn.utils import shuffle
import scipy.sparse
from tqdm.auto import tqdm


# In[2]:


# Make folder first!!
DATA_PATH = './data/raw/'
VAL_PATH='./result/val/'
TEST_PATH = './result/test/'
MODEL_PATH = './model/'


# In[5]:


train_data = pd.read_json(os.path.join(DATA_PATH,'train.json'), encoding='utf-8')
song_data = pd.read_json(os.path.join(DATA_PATH, 'song_meta.json'), encoding='utf-8')
valid_data = pd.read_json(os.path.join(DATA_PATH, 'val.json'), encoding='utf-8')
test_data = pd.read_json(os.path.join(DATA_PATH, 'test.json'), encoding='utf-8')
with open(os.path.join(DATA_PATH,'genre_gn_all.json'), encoding='utf-8') as json_file:
    genres = json.load(json_file)

song_data.index = song_data['id']
song_data.drop('id',inplace=True,axis=1)
song_data.loc[168071,"issue_date"] = 20010930
song_data["issue_year"] = (song_data["issue_date"]/10000).astype(int)

train_data["updt_year"] = pd.to_datetime(train_data["updt_date"]).dt.year
valid_data["updt_year"] = pd.to_datetime(valid_data["updt_date"]).dt.year
test_data["updt_year"] = pd.to_datetime(test_data["updt_date"]).dt.year

new_data = pd.concat([train_data,test_data,valid_data])

plylist_title = new_data["plylst_title"].apply(lambda x: x.split(' ')).explode().value_counts()

relevant_title_word = set(plylist_title[plylist_title>1].index)

def clean_name_list(val):
    splitted_string = val["plylst_title"].split(' ')
    new_val = []
    for nm in splitted_string:
        if nm in relevant_title_word:
            new_val.append('f_' + str(nm))
    new_val.append('y_'  + str(val["updt_year"]))
    new_val.append('p_' + str(int(np.log(val["like_cnt"] + 1))))
    return new_val

train_data["playlist_features"] = train_data.apply(clean_name_list,axis=1)
valid_data["playlist_features"] = valid_data.apply(clean_name_list,axis=1)
test_data["playlist_features"] = test_data.apply(clean_name_list,axis=1)

songs_count = new_data["songs"].explode().value_counts()

clean_songs = set(songs_count[songs_count>1].index)

def clean_song_list(val):
    new_val = []
    for son in val:
        if son in clean_songs:
            new_val.append(son)
    return new_val

train_data["songs"] = train_data["songs"].apply(clean_song_list)
valid_data["songs"] = valid_data["songs"].apply(clean_song_list)
test_data["songs"] = test_data["songs"].apply(clean_song_list)

tags_count = new_data["tags"].explode().value_counts()

clean_tags = set(tags_count[tags_count>1].index)

def clean_tag_list(val):
    new_val = []
    for tg in val:
        if tg in clean_tags:
            new_val.append(tg)
    return new_val

train_data["tags"] = train_data["tags"].apply(clean_tag_list)
valid_data["tags"] = valid_data["tags"].apply(clean_tag_list)
test_data["tags"] = test_data["tags"].apply(clean_tag_list)

unique_genre_list = []
def clean_genre_list(val):
    new_val = {}
    for son in val:
        for gen in song_data.loc[son,"song_gn_dtl_gnr_basket"]:
            if gen not in unique_genre_list:
                unique_genre_list.append(gen)
            if gen not in new_val:
                new_val[gen] = 0
            new_val[gen] += 1
        for gen in song_data.loc[son,"song_gn_gnr_basket"]:
            if gen not in unique_genre_list:
                unique_genre_list.append(gen)
            if gen not in new_val:
                new_val[gen] = 0
            new_val[gen] += 1
    if len(new_val)==0:
        return {}
    max_value = max(new_val.values())
    min_value = min(new_val.values())
    diff = max_value - min_value
    for key in new_val:
        if diff==0:
            new_val[key] = 1.0
            continue
        new_val[key] = 0.65 + (new_val[key]-min_value)*0.35/diff
    return new_val


train_data["genre_features"] = train_data["songs"].apply(clean_genre_list)
valid_data["genre_features"] = valid_data["songs"].apply(clean_genre_list)
test_data["genre_features"] = test_data["songs"].apply(clean_genre_list)


# In[6]:


song_data["issue_year"] = (song_data["issue_date"]/10000).astype(int)
song_data["issue_month"] = ((song_data["issue_date"]/100).astype(int))%100
song_data["issue_day"] = (song_data["issue_date"]%100).astype(int)

song_data.loc[(song_data["issue_day"]>31) | (song_data["issue_day"]<1),"issue_day"] = 1 
song_data.loc[(song_data["issue_month"]>12) | (song_data["issue_month"]<1),"issue_month"] = 1 
song_data.loc[(song_data["issue_year"]>2020) | (song_data["issue_year"]<1900),"issue_year"] = 1900 

song_data["corrected_date"] = song_data["issue_year"]*10000 + song_data["issue_month"]*100 + song_data["issue_day"]
song_data["year_month"] = song_data["issue_year"]*100 + song_data["issue_month"]


# In[7]:


#02. song, tags 맵핑 테이블 생성
# songs
tr_songs = list(train_data['songs'])
te_songs = list(test_data['songs'])
vl_songs = list(valid_data['songs'])
songs = tr_songs + te_songs + vl_songs
songs = list(set(list(np.hstack(songs))))

# # tags
tr_tags = list(train_data['tags'])
te_tags = list(test_data['tags'])
vl_tags = list(valid_data['tags'])
tags = tr_tags + te_tags + vl_tags
tags = list(set(list(np.hstack(tags))))

# # features
tr_features = list(train_data['playlist_features'])
te_features = list(test_data['playlist_features'])
vl_features = list(valid_data['playlist_features'])
features = tr_features + te_features + vl_features
features = list(set(list(np.hstack(features))))


# In[8]:


# codes
# codes = songs + tags

# song mapping dictionary
idx = 0
code_map = {}
for i, song_id in enumerate(songs):
    if song_id != '':
        code_map[song_id] = idx
        idx += 1
n_song = idx
print(n_song)

# addition tag mapping dictionary 
for i, tag_id in enumerate(tags):
    if tag_id != '':
        code_map[tag_id] = idx
        idx += 1

n_song_tag = idx
print('idx = ',n_song_tag)
for i, feature_id in enumerate(features):
    if feature_id != '':
        code_map[feature_id] = idx
        idx += 1

print('idx2 = ',idx)
for gen in unique_genre_list:
    code_map[gen] = idx
    idx += 1
    
n_codes = idx
print(n_codes)

# only tag mapping dictionary 
idx = 0
tag_map = {}
for i, tag_id in enumerate(tags):
    if tag_id != '':
        tag_map[tag_id] = idx
        idx += 1
n_tags = idx
print(n_tags)


# In[9]:


# code id to code dictionary 
code_mapi = {}
code_mapi = {value: key for key, value in code_map.items()}
# tag id to song dictionary 
tag_mapi = {}
tag_mapi = {value: key for key, value in tag_map.items()}
#%% transform to CSR


# In[10]:


def trans_csr(matrix1,matrix2,matrix3,matrix4, mapping):
    cols = []
    rows = []
    datas = []
    for i in range(len(matrix1)):
        k = [] + matrix1[i] + matrix2[i]
        col2 = []
        row2 = []
        data2 = []
        for code in k :
            if code != '' :
                col  = [mapping[code]]
                row  = [i]
                data = [1]
            col2.extend(col)
            row2.extend(row)
            data2.extend(data)
        for code in matrix3[i] :
            if code != '' :
                new_code = code
                col  = [mapping[new_code]]
                row  = [i]
                data = [1]
            col2.extend(col)
            row2.extend(row)
            data2.extend(data)
        for code in matrix4[i]:
            col  = [mapping[code]]
            row  = [i]
            data = [matrix4[i][code]]
            col2.extend(col)
            row2.extend(row)
            data2.extend(data)
        np.array(cols.extend(col2))
        np.array(rows.extend(row2))
        np.array(datas.extend(data2))
    return cols, rows, datas


# In[11]:


tr_col, tr_row, tr_data = trans_csr(matrix1 = tr_songs, matrix2 = tr_tags, matrix3 = tr_features, matrix4=list(train_data["genre_features"]), mapping = code_map)
te_col, te_row, te_data = trans_csr(matrix1 = te_songs, matrix2 = te_tags, matrix3 = te_features, matrix4=list(test_data["genre_features"]), mapping = code_map)
vl_col, vl_row, vl_data = trans_csr(matrix1 = vl_songs, matrix2 = vl_tags, matrix3 = vl_features, matrix4=list(valid_data["genre_features"]), mapping = code_map)


# In[12]:


size_mat = n_codes
tr_csr = csr_matrix((tr_data, (tr_row, tr_col)) , shape = (len(tr_features), size_mat) , dtype=int)
te_csr = csr_matrix((te_data, (te_row, te_col)) , shape = (len(te_features), size_mat) , dtype=int)
vl_csr = csr_matrix((vl_data, (vl_row, vl_col)) , shape = (len(vl_features), size_mat) , dtype=int)


# In[13]:


print(tr_csr.shape)
print(te_csr.shape)
print(vl_csr.shape)


# In[14]:


r = scipy.sparse.vstack([tr_csr, te_csr, vl_csr])
r


# In[15]:


# %%
als_model = ALS(factors=3072, regularization=0.08)
als_model.fit(r.T*50.0)


# In[17]:


import pickle
import gzip
import _pickle as cPickle
with gzip.open('./model/als_model-final-song-{}.gzip'.format(3072), 'wb') as f:
        cPickle.dump(als_model, f, pickle.HIGHEST_PROTOCOL)


# In[18]:


#%% item model과 tag model 로 분리 
song_model = ALS()
# tag_model = ALS()
#%% 
song_model.user_factors = als_model.user_factors
# tag_model.user_factors = als_model.user_factors

song_model.item_factors = als_model.item_factors[:n_song]
# tag_model.item_factors = als_model.item_factors[n_song:n_song_tag]
#%%
r_song_rec_csr = r[:, :n_song]
# r_tag_rec_csr = r[:, n_song:n_song_tag]
#%%
tr_id = train_data['id']
te_id = test_data['id']
vl_id = valid_data['id']
#%%


# In[26]:


songs_issue_dates = pd.DataFrame(pd.to_datetime(song_data["corrected_date"],format='%Y%m%d',errors='coerce'))
valid_data["updt_date"] = pd.to_datetime(valid_data["updt_date"])
test_data["updt_date"] = pd.to_datetime(test_data["updt_date"])


# In[20]:


songs_issue_dates["mappings"] = songs_issue_dates.index.map(lambda x: code_map[x] if x in code_map else -1)
songs_issue_dates =  songs_issue_dates[songs_issue_dates['mappings']!=-1]
songs_issue_dates.index = songs_issue_dates["mappings"]
songs_issue_dates.drop("mappings",axis=1,inplace=True)


# In[21]:


def clean_song_recommendations(u, vid, df):
    count = 0
    recommended_songs = []
    
    song_rec = song_model.recommend(u, r_song_rec_csr, N=10000)

    for song in song_rec:
        if song[0] in code_mapi and songs_issue_dates.loc[song[0],"corrected_date"]< df.iloc[vid]["updt_date"]:
            reco_son = int(code_mapi[song[0]])
            recommended_songs.append(reco_son)
            count+=1
        if count==100:
            break
    return recommended_songs


# In[22]:


val_song_ret = []
vid = 0
for u in tqdm(range(125811, 148826)):
    song_rec = clean_song_recommendations(u,vid, valid_data)
#     tag_rec = clean_tag_recommendations(u,vid)
    vid+=1
    val_song_ret.append(song_rec)
#     tag_ret.append(tag_rec)


# In[24]:


# val song result
submission = []
vid=-1
for _id, rec in zip(vl_id, val_song_ret):
    vid+=1
    if vid%1000==0:
        print(vid)

    submission.append({
        "id": _id,
        "songs": rec[:100],
        "tags": []
    })
        
#%% json 파일로 저장
VAL_SONG_PATH = os.path.join(VAL_PATH, 'song_results.json')
with open(VAL_SONG_PATH, 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii = False)


# In[27]:


test_song_ret = []
tid = 0
for u in tqdm(range(115071, 125811)):
    song_rec = clean_song_recommendations(u,tid, test_data)
    tid+=1
    test_song_ret.append(song_rec)


# In[29]:


# test song result
submission = []
tid=-1
for _id, rec in zip(te_id, test_song_ret):
    tid+=1
    if tid%1000==0:
        print(tid)

    submission.append({
        "id": _id,
        "songs": rec[:100],
        "tags": []
    })
        
#%% json 파일로 저장
TEST_SONG_PATH = os.path.join(TEST_PATH, 'song_results.json')
with open(TEST_SONG_PATH, 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii = False)


# # TAG ALS

# In[30]:


import os 
import json
import pandas as pd
import numpy as np
import pickle
import itertools
import operator
import scipy.sparse
import gzip
import plylst_title

from tqdm import tqdm
from datetime import datetime
from arena_tool import remove_seen
from implicit.evaluation import  *
from implicit.als import AlternatingLeastSquares as ALS


# In[31]:


class baseline:
    def __init__(self):
        self.FILE_PATH = DATA_PATH
        self.train_data  = pd.read_json(os.path.join(self.FILE_PATH,'train.json') , typ = 'frame')
        self.test_data   = pd.read_json(os.path.join(self.FILE_PATH,'test.json') , typ = 'frame')
        self.valid_data  = pd.read_json(os.path.join(self.FILE_PATH,'val.json'), typ = 'frame')
        self.song_data   = pd.read_json(os.path.join(self.FILE_PATH, 'song_meta.json'), typ = 'frame')   
        self.gga         = pd.read_json(os.path.join(self.FILE_PATH, 'genre_gn_all.json'), typ = 'series')

        self.tags_before_2016 = ['CCM', 'JPOP',  'OST',  '가을', '겨울', '기분전환', '까페',
        '뉴에이지', '댄스', '드라이브', '락', '랩', '매장음악', '발라드', '밤', '봄',
        '비오는날', '사랑', '산책', '새벽', '설렘', '소울', '스트레스', '슬픔',
        '알앤비', '여름', '여행', '운동', '월드뮤직', '이별', '인디', '일렉',
        '잔잔한', '재즈', '추억', '클래식', '클럽', '트로트', '팝', '회상', '휴식', '힐링', '힙합']

        self.genre_gn_all = self.gga.to_dict()

baseline = baseline()

# parameter and data
train_data = baseline.train_data 
test_data  = baseline.test_data
valid_data = baseline.valid_data
song_data  = baseline.song_data
tags_before_2016    = baseline.tags_before_2016
genre_gn_all =  baseline.genre_gn_all


# In[32]:


# get song feature dictionary
def get_song_feature_dict(song_meta):
    song_artist ={}
    song_genre = {}
    song_time = {}
    song_genre_dtl = {}
    song_album = {}

    for i in range(len(song_meta)):
        song_album[song_meta.loc[i,'id']]     = song_meta.loc[i,'album_name']
        song_artist[song_meta.loc[i,'id']]    = song_meta.loc[i,'artist_name_basket']
        song_genre[song_meta.loc[i,'id']]     = song_meta.loc[i,'song_gn_gnr_basket']
        song_genre_dtl[song_meta.loc[i,'id']] = song_meta.loc[i,'song_gn_dtl_gnr_basket']
        song_time[song_meta.loc[i,'id']]      = song_meta.loc[i,'issue_date']

    return song_artist, song_genre, song_genre_dtl, song_album, song_time

song_artist, song_genre, song_genre_dtl, song_album, song_time = get_song_feature_dict(song_meta = song_data)

# song_feature extract 

train_data["updt_year"] = pd.to_datetime(train_data["updt_date"]).dt.year
valid_data["updt_year"] = pd.to_datetime(valid_data["updt_date"]).dt.year
test_data["updt_year"] = pd.to_datetime(test_data["updt_date"]).dt.year

new_data = pd.concat([train_data,test_data,valid_data])

plylist_title = new_data["plylst_title"].apply(lambda x: x.split(' ')).explode().value_counts()
relevant_title_word = set(plylist_title[plylist_title>1].index)

songs_count = new_data["songs"].explode().value_counts()
clean_songs = set(songs_count[songs_count>0].index)

tags_count = new_data["tags"].explode().value_counts()
clean_tags = set(tags_count[tags_count>0].index)

def clean_name_list(val):
    splitted_string = val["plylst_title"].split(' ')
    new_val = []
    for nm in splitted_string:
        if nm in relevant_title_word:
            new_val.append('f_' + str(nm))
    new_val.append('y_'  + str(val["updt_year"]))
    new_val.append('p_' + str(int(np.log(val["like_cnt"] + 1))))
    return new_val

train_data["playlist_features"] = train_data.apply(clean_name_list,axis=1)
valid_data["playlist_features"] = valid_data.apply(clean_name_list,axis=1)
test_data["playlist_features"] = test_data.apply(clean_name_list,axis=1)

def clean_tag_list(val):
    new_val = []
    for tg in val:
        if tg in clean_tags:
            new_val.append(tg)
    return new_val

train_data["tags"] = train_data["tags"].apply(clean_tag_list)
valid_data["tags"] = valid_data["tags"].apply(clean_tag_list)
test_data["tags"] = test_data["tags"].apply(clean_tag_list)

unique_genre_list = []
def clean_genre_list(val):
    new_val = {}
    for son in val:
        for gen in song_data.loc[son,"song_gn_dtl_gnr_basket"]:
            if gen not in unique_genre_list:
                unique_genre_list.append(gen)
            if gen not in new_val:
                new_val[gen] = 0
            new_val[gen] += 1
        for gen in song_data.loc[son,"song_gn_gnr_basket"]:
            if gen not in unique_genre_list:
                unique_genre_list.append(gen)
            if gen not in new_val:
                new_val[gen] = 0
            new_val[gen] += 1
    if len(new_val)==0:
        return {}
    max_value = max(new_val.values())
    min_value = min(new_val.values())
    diff = max_value - min_value
    for key in new_val:
        if diff==0:
            new_val[key] = 1.0
            continue
        new_val[key] = 0.65 + (new_val[key]-min_value)*0.35/diff
    return new_val

train_data["genre_features"] = train_data["songs"].apply(clean_genre_list)
valid_data["genre_features"] = valid_data["songs"].apply(clean_genre_list)
test_data["genre_features"] = test_data["songs"].apply(clean_genre_list)

# convert dataframe to list of dictionaries
train = train_data.to_dict('record')
test = test_data.to_dict('record')
val = valid_data.to_dict('record')

def song_feature_extract(data):
    for q in tqdm(data, leave = True, position = 0):
        extract_feature = []

        # like count Processing
        if q['like_cnt'] > 1000 :
            q['like_cnt_cat'] = 'Extreme'
        elif q['like_cnt'] > 100 :       
            q['like_cnt_cat'] = 'Heavy'                                 
        elif q['like_cnt'] > 10 :       
            q['like_cnt_cat'] = 'Medium'                                                     
        else :
            q['like_cnt_cat'] = 'Light'

        extract_feature.append(q['like_cnt_cat'])

        song_features = []
        song_cnt = len(q['songs'])
        tag_cnt = len(q['tags'])
        
        # plylst title processing
        title_word = []
        title = q['plylst_title'].split()
        for word in title:
            if word in relevant_title_word:
                title_word.append('f_' + str(word))
        
        q['plylst_title_word'] = title_word

        if song_cnt > 0 :
            for song in q['songs'] :
                artist    = song_artist[song]
                album     = [song_album[song]]
                genre     = [genre_gn_all[x] for x in song_genre[song] if x in genre_gn_all]
                genre_dtl = [genre_gn_all[x] for x in song_genre_dtl[song] if x in genre_gn_all]
                genre_tot = [str(x) +"/" + str(y) for x in genre for y in genre_dtl]
                feature = artist + album + genre_tot            
                song_features.extend(feature)   

            song_features_status = pd.value_counts(song_features)/song_cnt

            give_feature = []
            for i in range(len(song_features_status)):

                if song_features_status.values[i] > 0.5:
                    give_feature.append(song_features_status.index[i])

            extract_feature.extend(give_feature)

        time = pd.to_datetime(q['updt_date'])
        year    = [str(time.year) + '년']
        month   = [str(time.month) + '월']
        day     = [str(time.day) + '일']
        hour    = [str(time.hour) + '시']
        weekday = [str(time.weekday()) + '요일']

        q['extract_feature'] = extract_feature + year + month + day + hour + weekday

#
song_feature_extract(data = train)
song_feature_extract(data = test)
song_feature_extract(data = val)

train_data = pd.DataFrame(train)
test_data = pd.DataFrame(test)
valid_data = pd.DataFrame(val)
tot_df = pd.concat([train_data, test_data, valid_data])

#
tot_df['genre_features'] = tot_df['genre_features'].apply(lambda x : list(x))
train_data['genre_features'] = train_data['genre_features'].apply(lambda x : list(x))
test_data['genre_features']  = test_data['genre_features'].apply(lambda x : list(x))
valid_data['genre_features'] = valid_data['genre_features'].apply(lambda x : list(x))


# In[33]:


#%% transform to CSR
def trans_csr(matrix1,matrix2,matrix3, matrix4, matrix5, mapping):
    cols = []
    rows = []
    datas = []
    for i in range(len(matrix1)):
        k = [] + matrix1[i] + matrix2[i] + matrix3[i] + matrix4[i] + matrix5[i]
        col2 = []
        row2 = []
        data2 = []
        for code in k :
            if code != '' :
                col  = [mapping[code]]
                row  = [i]
                data = [1]
            col2.extend(col)
            row2.extend(row)
            data2.extend(data)
        np.array(cols.extend(col2))
        np.array(rows.extend(row2))
        np.array(datas.extend(data2))
    return cols, rows, datas
    
# songs
songs = tot_df['songs']
songs = list(set(list(np.hstack(songs))))

# tags
tags = tot_df['tags']
tags = list(set(list(np.hstack(tags))))

# extract features
features = tot_df['extract_feature']
features = list(set(list(np.hstack(features))))   

# playlist title word
title_word = tot_df['plylst_title_word']
title_word = list(set(list(np.hstack(title_word))))   

# genre feature
genre_features = tot_df['genre_features']
genre_features = list(set(list(np.hstack(genre_features))))   

# codes
codes = songs + tags + features + title_word +genre_features

# song mapping dictionary
idx = 0
code_map = {}
for i, song_id in enumerate(songs):
    if song_id != '':
        code_map[song_id] = idx
        idx += 1
n_song = idx
print(n_song)

# addition tag mapping dictionary 
for i, tag_id in enumerate(tags):
    if tag_id != '':
        code_map[tag_id] = idx
        idx += 1
n_tag = idx
print(n_tag)

# addition feature mapping dictionary 
for i, fe_id in enumerate(features):
    if fe_id != '':
        code_map[fe_id] = idx
        idx += 1
n_feature = idx
print(n_feature)

# addition plylst title word mapping dictionary 
for i, fe_id in enumerate(title_word):
    if fe_id != '':
        code_map[fe_id] = idx
        idx += 1
n_title_word = idx
print(n_title_word)

# addition genre feature mapping dictionary 
for i, ge_id in enumerate(genre_features):
    if ge_id != '':
        code_map[ge_id] = idx
        idx += 1
n_genre_feature = idx
print(n_genre_feature)

# only tag mapping dictionary 
idx = 0
tag_map = {}
for i, tag_id in enumerate(tags):
    if tag_id != '':
        tag_map[tag_id] = idx
        idx += 1
n_tags = idx
print(n_tags)

# code id to code dictionary 
code_mapi = {}
code_mapi = {value: key for key, value in code_map.items()}
# tag id to song dictionary 
tag_mapi = {}
tag_mapi = {value: key for key, value in tag_map.items()}

#%%
tot_songs = list(tot_df['songs'])
tot_tags  = list(tot_df['tags'])
tot_extract_feature =  list(tot_df['extract_feature'])
tot_title_word =  list(tot_df['plylst_title_word'])
tot_genre_feature =  list(tot_df['genre_features'])

#%%
tot_col, tot_row, tot_data = trans_csr(matrix1 = tot_songs, 
matrix2 = tot_tags, 
matrix3 = tot_extract_feature, 
matrix4 = tot_title_word, 
matrix5 = tot_genre_feature, 
mapping = code_map)

tot_csr = scipy.sparse.csr_matrix((tot_data, (tot_row, tot_col)) , dtype=int)

r = tot_csr


# In[34]:


als_tag = ALS(factors=300, regularization=0.08, iterations=30)
als_tag.fit(item_users= r.T*40)


# In[35]:


n_train = len(train_data)
n_test  = len(test_data) + len(train_data)
n_val   = len(test_data) + len(train_data) + len(valid_data)
print(n_train)
print(n_test)
print(n_val)
#%%
# tag model 분리 
als_tag_only = ALS()
als_tag_only.user_factors = als_tag.user_factors
als_tag_only.item_factors = als_tag.item_factors[n_song:n_tag]
r_tag_rec_csr = r[:, n_song:n_tag]

tot_id = tot_df['id']

# year wise tag rule
train_data["updt_year"] = pd.to_datetime(train_data["updt_date"]).dt.year
tags_with_min_year = train_data[["tags","updt_year"]].explode("tags").groupby("tags").min()

year_wise_tags_list = {}
for year in range(2004,2020):
    year_tags = list(tags_with_min_year[tags_with_min_year["updt_year"]<=year].index)
    year_tag_ids = []
    for tag in year_tags:
        if tag not in tag_map:
            continue
        year_tag_ids.append(tag_map[tag])
    year_wise_tags_list[year] = year_tag_ids.copy()

# 2004년의 경우 tag가 4개밖에 없어 error 발생 하므로 2005년 tag를 준용하여 사용한다.
year_wise_tags_list[2004] = year_wise_tags_list[2005]

tot_df['updt_date'] = pd.to_datetime(tot_df['updt_date'])


# In[38]:


#  현재까지 가장 잘나온 val song 결과 활용 
with open(VAL_SONG_PATH, encoding="utf-8") as f:
    result_song = json.load(f)

submission2 = []
for u in tqdm(range(n_test, n_val), position=0, leave=True):
    song_ret = []
    tag_ret = []

    if tot_df.iloc[u]["updt_date"] < pd.to_datetime('20160101', format='%Y%m%d', errors='coerce'):
        tag_rec = als_tag_only.rank_items(u, r_tag_rec_csr, year_wise_tags_list[year])
    else:
        tag_rec = als_tag_only.recommend(u, r_tag_rec_csr, N=20, filter_already_liked_items=True)

    for tag_id in tag_rec:
        if tag_id[0] in tag_mapi:
            tag = tag_mapi[tag_id[0]]
            tag_ret.append(tag)

    submission2.append({
    "id": val[u-n_test]["id"],
    "songs": result_song1[u-n_test]["songs"],
    "tags":  remove_seen(val[u-n_test]['tags'] ,tag_ret)[:10]
    })


# In[39]:


VAL_ALS_PATH = os.path.join(VAL_PATH, 'als_result.json')
with open(VAL_ALS_PATH , 'w', encoding='utf-8') as f:
    json.dump(submission2, f, ensure_ascii = False)


# In[40]:


#  현재까지 가장 잘나온 test song 결과 활용 
with open(TEST_SONG_PATH, encoding="utf-8") as f:
    result_song = json.load(f)
    
submission3 = []
for u in tqdm(range(n_train, n_test), position=0, leave=True):
    song_ret = []
    tag_ret = []

    if tot_df.iloc[u]["updt_date"] < pd.to_datetime('20160101', format='%Y%m%d', errors='coerce'):
        tag_rec = als_tag_only.rank_items(u, r_tag_rec_csr, year_wise_tags_list[year])
    else:
        tag_rec = als_tag_only.recommend(u, r_tag_rec_csr, N=20, filter_already_liked_items=True)

    for tag_id in tag_rec:
        if tag_id[0] in tag_mapi:
            tag = tag_mapi[tag_id[0]]
            tag_ret.append(tag)

    submission3.append({
    "id": test[u-n_train]["id"],
    "songs": result_song[u-n_train]["songs"],
    "tags":  remove_seen(val[u-n_train]['tags'] ,tag_ret)[:10]
    })


# In[41]:


TEST_ALS_PATH = os.path.join(TEST_PATH, 'als_result.json')
with open(TEST_ALS_PATH, 'w', encoding='utf-8') as f:
    json.dump(submission3, f, ensure_ascii = False)


# # Tag w2v ensemble

# In[3]:


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


# In[4]:


FILE_PATH = DATA_PATH
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

tags_before_2016 = ['CCM', 'JPOP',  'OST',  '가을', '겨울', '기분전환', '까페',
'뉴에이지', '댄스', '드라이브', '락', '랩', '매장음악', '발라드', '밤', '봄',
'비오는날', '사랑', '산책', '새벽', '설렘', '소울', '스트레스', '슬픔',
'알앤비', '여름', '여행', '운동', '월드뮤직', '이별', '인디', '일렉',
'잔잔한', '재즈', '추억', '클래식', '클럽', '트로트', '팝', '회상', '휴식', '힐링', '힙합']
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

#%%
tot = train + test + val
song_dic = {}
tag_dic = {}
for q in tqdm(tot):
    song_dic[q['id']] = q['songs']
    tag_dic[q['id']] = q['tags']


# In[11]:


# model load 
def hash(astring):
   return ord(astring[0])

with open(os.path.join(MODEL_PATH,'0007ky.p2v'), 'rb') as f:
    p2v_model = pickle.load(f)

with open(os.path.join(MODEL_PATH,'0007ky.w2v'), 'rb') as f:
    w2v_model = pickle.load(f)


# In[12]:


# tag rule for before 2016
# only tag mapping dictionary 
tags = new_data['tags']
tags = list(set(list(np.hstack(tags))))
#%%
idx = 0
tag_map = {}
for i, tag_id in enumerate(tags):
    if tag_id != '':
        tag_map[tag_id] = idx
        idx += 1
n_tags = idx
print(n_tags)

# tag id to song dictionary 
tag_mapi = {}
tag_mapi = {value: key for key, value in tag_map.items()}

# %%
train_data["updt_year"] = pd.to_datetime(train_data["updt_date"]).dt.year
tags_with_min_year = train_data[["tags","updt_year"]].explode("tags").groupby("tags").min()

#
year_wise_tags_list = {}
year_wise_tags_list2 = {}
for year in range(2004,2020):
    year_tags = list(tags_with_min_year[tags_with_min_year["updt_year"]<=year].index)
    year_tag_ids = []
    for tag in year_tags:
        if tag not in tag_map:
            continue
        year_tag_ids.append(tag_map[tag])
    year_wise_tags_list[year] = year_tag_ids.copy()
    year_wise_tags_list2[year] = [tag_mapi[x] for x in year_wise_tags_list[year]]

year_wise_tags_list2[2004] = year_wise_tags_list2[2005]


# In[15]:


# for valid result
VAL_ALS_PATH = os.path.join(VAL_PATH, 'als_result.json')
VAL_SONG_PATH = os.path.join(VAL_PATH, 'song_results.json')

# from als tag model
with open(VAL_ALS_PATH, encoding="utf-8") as f:
    result_tag = json.load(f)
# from als song model
with open(VAL_SONG_PATH, encoding="utf-8") as f:
    result_song = json.load(f)

target = val
submission = []
for q in tqdm(target, position= 0, leave = True):
    tmp_vec = 0
    if len(q['repr'])>0:
        for word in q['repr']:
            try: 
                tmp_vec += w2v_model.wv.get_vector(word)
            except KeyError:
                pass
    else :
        pass
    
    get_song = []
    get_tag  = []

    if type(tmp_vec) != int:
        most_id = [x[0] for x in p2v_model.wv.most_similar([tmp_vec] , topn = 10)]    
    for ID in most_id:
        get_song += song_dic[int(ID)]
        get_tag += tag_dic[int(ID)]

# 2016 before tag rule
    time = datetime.strptime(q['updt_date'], '%Y-%m-%d %H:%M:%S.%f')
    year = time.year

    if year < 2016:
        get_tag = [word for word in get_tag if word in year_wise_tags_list2[year]]
        
    get_song = list(pd.value_counts(get_song)[:200].index)    
    get_tag = list(pd.value_counts(get_tag)[:20].index)    

    submission.append({
        "id": q["id"],
        "songs": remove_seen(q["songs"], get_song)[:100],
        "tags": remove_seen(q["tags"], get_tag)[:3]
    })

    for n, q in enumerate(submission):
        submission[n]['songs'] = result_song[n]['songs']
        if len(q['tags'])!=10:
            submission[n]['tags'] += remove_seen(q['tags'], result_tag[n]['tags'])[:10-len(q['tags'])]  

VAL_FINAL = os.path.join(VAL_PATH, 'results.json')
with open(VAL_FINAL, 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii = False)


# In[16]:


# for test result
TEST_ALS_PATH = os.path.join(TEST_PATH, 'als_result.json')
TEST_SONG_PATH = os.path.join(TEST_PATH, 'song_results.json')
# tag als result
with open(TEST_ALS_PATH, encoding="utf-8") as f:
    result_tag = json.load(f)
#  song als result 
with open(TEST_SONG_PATH, encoding="utf-8") as f:
    result_song = json.load(f)
#%%
target = test
submission4 = []
for q in tqdm(target, position= 0, leave = True):
    tmp_vec = 0
    if len(q['repr'])>0:
        for word in q['repr']:
            try: 
                tmp_vec += w2v_model.wv.get_vector(word)
            except KeyError:
                pass
    else :
        pass
    
    get_song = []
    get_tag  = []

    if type(tmp_vec) != int:
        most_id = [x[0] for x in p2v_model.wv.most_similar([tmp_vec] , topn = 10)]    
    for ID in most_id:
        get_song += song_dic[int(ID)]
        get_tag += tag_dic[int(ID)]

# 2016 before tag rule
    time = datetime.strptime(q['updt_date'], '%Y-%m-%d %H:%M:%S.%f')
    year = time.year

    if year < 2016:
        get_tag = [word for word in get_tag if word in year_wise_tags_list2[year]]

    get_song = list(pd.value_counts(get_song)[:200].index)    
    get_tag = list(pd.value_counts(get_tag)[:20].index)    

    submission4.append({"id": q["id"], 
                        "songs": remove_seen(q["songs"], get_song)[:100],
                        "tags": remove_seen(q["tags"], get_tag)[:3]
    })
    
    for n, q in enumerate(submission4):
        submission4[n]['songs'] = result_song[n]['songs']
        if len(q['tags'])!=10:
            submission4[n]['tags'] += remove_seen(q['tags'], result_tag[n]['tags'])[:10-len(q['tags'])]  

TEST_FINAL = os.path.join(TEST_PATH, 'results.json')
with open(TEST_FINAL, 'w', encoding='utf-8') as f:
    json.dump(submission4, f, ensure_ascii = False)


# In[ ]:





# In[ ]:




