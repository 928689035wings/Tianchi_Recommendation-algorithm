import pandas as pd

import numpy as np

train = pd.read_csv('H:\\pythonchengx_u\\Tianchiantai\\dianshangtuijian\\Antai_AE_round1_train_20190626.csv')
test = pd.read_csv('H:\\pythonchengx_u\\Tianchiantai\\dianshangtuijian\\Antai_AE_round1_test_20190626.csv')

all_data = pd.concat([train,test])
all_data = all_data.sort_values(by=['buyer_admin_id', 'irank'], ascending=[True, True])
from sklearn.model_selection import train_test_split

df_ratings_train, df_ratings_test= train_test_split(all_data,
                                                    stratify=all_data['buyer_admin_id'],
                                                    random_state = 15688,
                                                    test_size=0.30)

print("Number of training data: "+str(len(df_ratings_train)))
print("Number of test data: "+str(len(df_ratings_test)))


# 为每个用户生成列表 用户的每一个item看做一个词，用户所有的item 在一个句子里
def get_preprocessing(df_):
    df = df_.copy()   
    df['hour']  = df['create_order_time'].apply(lambda x:int(x[11:13]))
    df['day']   = df['create_order_time'].apply(lambda x:int(x[8:10]))
    df['month'] = df['create_order_time'].apply(lambda x:int(x[5:7]))
    df['year']  = df['create_order_time'].apply(lambda x:int(x[0:4]))
    df['date']  = (df['month'].values - 7) * 31 + df['day']    
    del df['create_order_time']    
    return df

def splitter(df):

    df['item_id'] = df['item_id'].astype('str')
    gp_user_like = df.groupby(['buyer_admin_id','date'])
    return ([gp_user_like.get_group(gp)['item_id'].tolist() for gp in gp_user_like.groups])

all_data = get_preprocessing(all_data)
splitted_items = splitter(all_data)

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
assert gensim.models.word2vec.FAST_VERSION > -1

import random

# for item_list in splitted_items:

#     random.shuffle(item_list)

from gensim.models import Word2Vec
import datetime
start = datetime.datetime.now()

model = Word2Vec(sentences = splitted_items, # We will supply the pre-processed list of moive lists to this parameter
                 iter = 10, # epoch
                 min_count = 1, # a movie has to appear more than 10 times to be keeped
                 size = 200, # size of the hidden layer
                 workers = 4, # specify the number of threads to be used for training
                 sg = 1, # Defines the training algorithm. We will use skip-gram so 1 is chosen.
                 hs = 0, # Set to 0, as we are applying negative sampling.
                 negative = 5, # If > 0, negative sampling will be used. We will use a value of 5.
                 window = 10)

print("Time passed: " + str(datetime.datetime.now()-start))
#Word2Vec.save('item2vec_20180327')

model.save('H:\\pythonchengx_u\\Tianchiantai\\dianshangtuijian\\item2vec_8_9_5')