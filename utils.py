import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Queue
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import pickle
# import pygeohash
# from geopy.distance import geodesic
# from concurrent.futures import ThreadPoolExecutor

def sample_function(user, user_train, geo_train, dis_train, usernum, itemnum, geonum, disnum, batch_size, maxlen, result_queue, SEED):
    
    # sampler for batch generation
    def random_neq(l, r, s):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t
    
    def sample():

        data_idx = np.random.randint(0, user_train.size)
        while len(user_train[data_idx]) <= 1: data_idx = np.random.randint(0, user_train.size)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        geo = np.zeros([maxlen], dtype=np.int32)
        geo_pos = np.zeros([maxlen], dtype=np.int32)
        dis = np.zeros([maxlen], dtype=np.int32)
        dis_pos = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[data_idx][-1]
        nxt_geo = geo_train[data_idx][-1]
        nxt_dis = dis_train[data_idx][-1]
        
        idx = maxlen - 1
        ts = set(user_train[data_idx])
        for i in reversed(user_train[data_idx][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)  #随机找一个其他的
            nxt = i
            idx -= 1
            if idx == -1: break
        # Process Geo
        idx = maxlen - 1
        for j in reversed(geo_train[data_idx][:-1]):
            geo[idx] = j
            geo_pos[idx] = nxt_geo
            nxt_geo = j
            idx -= 1
            if idx == -1: break
        # Process Dis
        idx = maxlen - 1
        for j in reversed(dis_train[data_idx][:-1]):
            dis[idx] = j
            dis_pos[idx] = nxt_dis
            dis_geo = j
            idx -= 1
            if idx == -1: break

        return (user[data_idx], seq, pos, neg, geo, geo_pos, dis, dis_pos)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, U, User, Geo, Dis, usernum, itemnum, geonum, disnum, batch_size=64, maxlen=10, n_workers=1, seed=42):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(U, User, Geo, Dis, usernum, itemnum, geonum, disnum, 
                                                      batch_size, maxlen, self.result_queue, seed)))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def data_partition(args):
    """
    Loads and partitions the dataset from pickle files.

    Returns:
        A list containing all dataset components for training, validation, and testing.
    """
    base_path = Path(args.dataset_dir) / args.city
    student_data_path = base_path / f"{args.city}_student_data.pkl"
    dict_path = base_path / f"{args.city}_dict.pkl"
    
    df = pickle.load(open(student_data_path, 'rb'))
    entity_dict = pickle.load(open(dict_path, 'rb'))

    # Unpack data into a list for consistent return format
    return [
        df['user'], df['user_train'], df['user_val'], df['user_test'],
        df['geo_train'], df['geo_val'], df['geo_test'],
        df['dis_train'], df['dis_val'], df['dis_test'],
        entity_dict['usernum'], entity_dict['itemnum'],
        entity_dict['geonum'], entity_dict['disnum']
    ]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, gcn_model, dataset, args, batch_size=128):
    [user, train, valid, test, geo_train, geo_val, geo_test, dis_train, dis_val, dis_test, usernum, itemnum, geonum, disnum] = copy.deepcopy(dataset)

    NDCG1 = 0.0
    NDCG5 = 0.0
    NDCG10 = 0.0
    NDCG20 = 0.0
    NDCG50 = 0.0
    HT1 = 0.0
    HT5 = 0.0
    HT10 = 0.0
    HT20 = 0.0
    HT50 = 0.0
    valid_user = 0.0

    data_idx = range(0, train.size)
    
    for batch_start in tqdm(range(0, len(data_idx), batch_size)):
        batch_end = min(batch_start + batch_size, len(data_idx))
        batch_users = data_idx[batch_start:batch_end]

        seq_batch = np.zeros([len(batch_users), args.maxlen], dtype=np.int32)
        geo_batch = np.zeros([len(batch_users), args.maxlen], dtype=np.int32)
        dis_batch = np.zeros([len(batch_users), args.maxlen], dtype=np.int32)
        item_idx_batch = []
        user_batch = []

        for i, u in enumerate(batch_users):
            if len(train[u]) < 1 or len(test[u]) < 1: continue

            idx = args.maxlen - 1
            seq_batch[i][idx] = valid[u][0]
            geo_batch[i][idx] = geo_val[u][0]
            dis_batch[i][idx] = dis_val[u][0]

            idx -= 1
            for t in reversed(train[u]):
                seq_batch[i][idx] = t
                idx -= 1
                if idx == -1: break


            idx = args.maxlen - 1
            idx -= 1
            for t in reversed(geo_train[u]):
                geo_batch[i][idx] = t
                idx -= 1
                if idx == -1: break

            idx = args.maxlen - 1
            idx -= 1
            for t in reversed(dis_train[u]):
                dis_batch[i][idx] = t
                idx -= 1
                if idx == -1: break

            rated = set(train[u] + valid[u])
            rated.add(0)
            item_idx = [test[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            item_idx_batch.append(item_idx)
            user_batch.append(user[u])

        if len(user_batch) == 0:
            continue

        user_batch = np.array(user_batch)
        item_idx_batch = np.array(item_idx_batch)

        predictions = -model.predict(
            torch.LongTensor(user_batch).to(args.device), 
            torch.LongTensor(seq_batch).to(args.device), 
            torch.LongTensor(geo_batch).to(args.device), 
            torch.LongTensor(dis_batch).to(args.device), 
            np.array(item_idx_batch)
        )

        
        for i in range(len(user_batch)):
            u = user_batch[i]
            rank = predictions[i].argsort().argsort()[0].item()
            
            valid_user += 1

            if rank < 1:
                NDCG1 += 1 / np.log2(rank + 2)
                HT1 += 1
            if rank < 5:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
            if rank < 10:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
            if rank < 20:
                NDCG20 += 1 / np.log2(rank + 2)
                HT20 += 1
            if rank < 50:
                NDCG50 += 1 / np.log2(rank + 2)
                HT50 += 1

    NDCG1 = NDCG1 / valid_user
    NDCG5 = NDCG5 / valid_user
    NDCG10 = NDCG10 / valid_user
    NDCG20 = NDCG20 / valid_user
    NDCG50 = NDCG50 / valid_user
    HT1 = HT1 / valid_user
    HT5 = HT5 / valid_user
    HT10 = HT10 / valid_user
    HT20 = HT20 / valid_user
    HT50 = HT50 / valid_user

    return [NDCG1, NDCG5, NDCG10, NDCG20, NDCG50], [HT1, HT5, HT10, HT20, HT50]

def evaluate_valid(model, gcn_model, dataset, args, batch_size=128):
    [user, train, valid, test, geo_train, geo_val, geo_test, dis_train, dis_val, dis_test, usernum, itemnum, geonum, disnum] = copy.deepcopy(dataset)

    NDCG1 = 0.0
    NDCG5 = 0.0
    NDCG10 = 0.0
    NDCG20 = 0.0
    NDCG50 = 0.0
    HT1 = 0.0
    HT5 = 0.0
    HT10 = 0.0
    HT20 = 0.0
    HT50 = 0.0
    valid_user = 0.0

    data_idx = list(range(len(train)))
    
    for batch_start in tqdm(range(0, len(data_idx), batch_size)):
        
        # if batch_start/batch_size == 150:
        #     print("150")

        batch_end = min(batch_start + batch_size, len(data_idx))
        batch_users = data_idx[batch_start:batch_end]

        seq_batch = np.zeros([len(batch_users), args.maxlen], dtype=np.int32)
        geo_batch = np.zeros([len(batch_users), args.maxlen], dtype=np.int32)
        dis_batch = np.zeros([len(batch_users), args.maxlen], dtype=np.int32)
        item_idx_batch = []
        user_batch = []

        for i, u in enumerate(batch_users):
            if len(train[u]) < 1 or len(valid[u]) < 1: 
                continue

            idx = args.maxlen - 1
            for t in reversed(train[u]):
                seq_batch[i][idx] = t
                idx -= 1
                if idx == -1: break

            idx = args.maxlen - 1
            for t in reversed(geo_train[u]):
                geo_batch[i][idx] = t
                idx -= 1
                if idx == -1: break

            idx = args.maxlen - 1
            for t in reversed(dis_train[u]):
                dis_batch[i][idx] = t
                idx -= 1
                if idx == -1: break

            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: 
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            item_idx_batch.append(item_idx)
            user_batch.append(user[u])

        if len(user_batch) == 0:
            continue

        user_batch = np.array(user_batch)
        item_idx_batch = np.array(item_idx_batch)

        predictions = -model.predict(
            torch.LongTensor(user_batch).to(args.device), 
            torch.LongTensor(seq_batch).to(args.device), 
            torch.LongTensor(geo_batch).to(args.device), 
            torch.LongTensor(dis_batch).to(args.device), 
            np.array(item_idx_batch)
        )
        
        for i in range(len(user_batch)):
            
            rank = predictions[i].argsort().argsort()[0].item()
            
            valid_user += 1

            if rank < 1:
                NDCG1 += 1 / np.log2(rank + 2)
                HT1 += 1
            if rank < 5:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
            if rank < 10:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
            if rank < 20:
                NDCG20 += 1 / np.log2(rank + 2)
                HT20 += 1
            if rank < 50:
                NDCG50 += 1 / np.log2(rank + 2)
                HT50 += 1
        # break
    NDCG1 /= valid_user
    NDCG5 /= valid_user
    NDCG10 /= valid_user
    NDCG20 /= valid_user
    NDCG50 /= valid_user
    HT1 /= valid_user
    HT5 /= valid_user
    HT10 /= valid_user
    HT20 /= valid_user
    HT50 /= valid_user

    return [NDCG1, NDCG5, NDCG10, NDCG20, NDCG50], [HT1, HT5, HT10, HT20, HT50]
