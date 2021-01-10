import pandas as pd
import random
import gc

random.seed(1)


train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',
                   dtype={'row_id': 'int64',
                          'timestamp': 'int64',
                          'user_id': 'int32',
                          'content_id': 'int16',
                          'content_type_id': 'int8',
                          'task_container_id': 'int16',
                          'user_answer': 'int8',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'float32',
                          'prior_question_had_explanation': 'boolean'}
                   )


max_timestamp_u = train[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
max_timestamp_u.columns = ['user_id', 'max_time_stamp']
MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()


def rand_time(max_time_stamp):
    interval = MAX_TIME_STAMP - max_time_stamp
    rand_time_stamp = random.randint(0,interval)
    return rand_time_stamp

max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.apply(rand_time)
train = train.merge(max_timestamp_u, on='user_id', how='left')
train['viretual_time_stamp'] = train.timestamp + train['rand_time_stamp']


del train['max_time_stamp']
del train['rand_time_stamp']
del max_timestamp_u
gc.collect()

train = train.sort_values(['viretual_time_stamp', 'row_id']).reset_index(drop=True)


val_size = 2500000

for cv in range(2):
    valid = train[-val_size:]
    train = train[:-val_size]
    valid[["row_id"]].to_pickle(f'cv{cv+1}_valid.pickle')