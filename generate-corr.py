!pip install datatable

import pandas as pd
import datatable as dt

import numpy as np
import sys

train_data = dt.fread("/input/riiid-test-answer-prediction/train.csv").to_pandas()

train_data = train_data[train_data.content_type_id == False]

idxs =train_data.groupby("content_id")["user_id"].count().nlargest(300).index

train_data =  train_data[train_data.content_id.isin(idxs)]

del train_data["row_id"]
del train_data["content_type_id"]
del train_data["task_container_id"]
del train_data["prior_question_elapsed_time"]
del train_data["prior_question_had_explanation"]
for i in idxs:
    train_data["q_"+str(i)] = 0
    train_data["q_"+str(i)] = train_data["q_"+str(i)].astype("bool")
    
    
cols = train_data.columns[5:]

cols_2_idx = {cols[i]:i for i in range(len(cols))}

tmp = train_data[cols].astype("bool").values

q_tmp = train_data[train_data.columns[:5]].values

idx_2_que = {i: int(cols[i][2:]) for i in range(len(cols))}

user_info = {}

from tqdm import tqdm

for i in tqdm(range(tmp.shape[0])):
    
    que_id = q_tmp[i,2]
    correct = q_tmp[i,-1]
    
    user_id = q_tmp[i, 1]
    
    if correct: 
        if user_info.get(user_id, -1) == -1:
            user_info[user_id] = {que_id: 1}
        else:
            user_info[user_id][que_id] = 1
            
    for col in cols:
        col_name = cols_2_idx[col]
        tmp[i, col_name] = user_info.get(user_id, {}).get(idx_2_que[col_name], 0)
        
        
train_data = pd.DataFrame(tmp, columns=cols)
heat= train_data.corr() #Will take time and a lot of memory

heat.to_pickle("heat_pickle")