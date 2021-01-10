import pandas as pd
heat = pd.read_pickle("heat_pickle")

import datatable as dt

train_data = dt.fread("/input/riiid-test-answer-prediction/train.csv").to_pandas()


train_data = train_data[train_data.content_type_id == False]
idxs =train_data.groupby("content_id")["user_id"].count().nlargest(300).index

cols = []
for i in idxs:
    cols.append("q_"+str(i))
    
    
idx_2_que = {i: int(cols[i][2:]) for i in range(len(cols))}

que_corr = {}
for i in cols:
    dct = heat[i].nlargest(6)[1:].to_dict()
    que_corr[int(i[2:])] = {int(j[2:]):dct[j] for j in dct}
    
    
del train_data

train_data = pd.read_pickle("/home/dfs/data_2")

train_data = train_data[["user_id","content_id","answered_correctly","task_container_id"]]

import numpy as np

train_values = train_data.values
train_idxs = train_data.index

history = {}
vals = np.zeros((train_data.shape[0], 6))

from tqdm import tqdm

for i in tqdm(range(train_values.shape[0])):

    que_id = train_values[i, 1]
    user_id = train_values[i, 0]
    correct = train_values[i, 2]

    if que_id in idxs:

        vals[i, 5] = 1
        
        correlated = que_corr[que_id]
        corr_keys = list(correlated.keys())
        
        user_history = history.get(user_id, {})
        
        for j in range(len(corr_keys)):
    
            if user_history.get(corr_keys[j], False):   #If user answered this question
                vals[i, j] = correlated[corr_keys[j]]   #Set the value to the degree of the correlation

        if correct == True: 

            if history.get(user_id, -1) == -1:
                history[user_id] = {}
            history[user_id][que_id] = True

            
train_values = np.hstack((train_values.astype("float64"), vals))

vals = pd.DataFrame(train_values)
vals.columns = ["user_id","content_id","answered_correctly","task_container_id", "corr_1","corr_2","corr_3","corr_4","corr_5","top_que"]

vals = vals.astype({"user_id":"int64","content_id":"int32","answered_correctly":"bool",
                    "task_container_id":"int32", "corr_1":"float16", "corr_2":"float16",
                    "corr_3":"float16","corr_4":"float16","corr_5":"float16","top_que":"bool"})

vals.index = train_idxs
vals.to_pickle("/home/dfs/corr")