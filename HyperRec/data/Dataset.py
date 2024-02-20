import pickle
import math

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict
import datetime
import operator

data_path = 'data/'

def filter_k_core_inters(inters, user_inter_threshold=5, item_inter_threshold=5):
    print(f"Filter K core: user {user_inter_threshold}, item {item_inter_threshold}")
    while True:
        user_count = {}
        item_count = {}
        for inter in inters:
            if inter[0] not in user_count:
                user_count[inter[0]] = 1
            else:
                user_count[inter[0]] += 1

            if inter[1] not in item_count:
                item_count[inter[1]] = 1
            else:
                item_count[inter[1]] += 1

        new_inters = []
        for inter in inters:
            if user_count[inter[0]] >= user_inter_threshold and \
                    item_count[inter[1]] >= item_inter_threshold:
                new_inters.append(inter)

        print(f"\tFilter: {len(inters)} inters to {len(new_inters)} inters")
        if len(new_inters) == len(inters):
            return new_inters
        inters = new_inters

def data_partition_neg(args):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    User_time = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_train_valid = {}
    user_test = {}
    neg_test = {}

    time_set_train = set()
    time_set_test = set()

    user_train_time = {}
    user_valid_time = {}
    user_train_valid_time = {}
    user_test_time = {}
    # assume user/item index starting from 1
    path_to_data = data_path + args.data + '/' + 'ratings_Beauty.csv'
    inters = []
    f = open(path_to_data, 'r')
    for uid, line in enumerate(f):
        u, i, r, t = line.rstrip().split(',')
        inters.append([u, i, t])
    new_iters = filter_k_core_inters(inters)
    uids, iids = {}, {}
    u, i = 1, 1
    for line in new_iters:
        if line[0] not in uids:
            uids[line[0]] = u
            u += 1
        if line[1] not in iids:
            iids[line[1]] = i
            i += 1
        ids_iter = []
    for line in new_iters:
        ids_iter.append([uids[line[0]], iids[line[1]], line[2]])

    # for Amazon    
    t_map = {1997:[1], 1998:[1], 1999:[1], 2000:[1], 2001:[1], 2002:[1], 2003:[1], 2004:[1], 2005:[1], 2006:[1], 2007:[1], 2008:[1], 2009:[2,3], 2010:[4,5], \
    2011:[6,7], 2012:[8,9], 2013:[10,11], 2014:[12,13], 2015:[14,15], 2016:[16,17], 2017:[18,19], 2018:[20]}

     
    m_map = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1}

    # with open(path_to_data, 'rb') as out:
    #     f = pickle.load(out)
    # f = open(path_to_data, 'r')
    f = ids_iter
    for line in f:
        [u, i, t] = line
        # u, i, t, d = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        year = int(datetime.datetime.fromtimestamp(int(t)).strftime("%Y")) # Day of the year as a decimal number [001,366]
        month = int(datetime.datetime.fromtimestamp(int(t)).strftime("%m"))

        
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)   

        temp_map = t_map[year]
        if len(temp_map) == 1:
            User_time[u].append(temp_map[0])
        else:
            User_time[u].append(temp_map[m_map[month]]) 


    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []

            user_train_time[user] = User_time[user]
            user_valid_time[user] = []
            user_test_time[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
            
            neg_test[user] = [User[user][-1]]

            user_train_time[user] = User_time[user][:-2]
            time_set_train.update(user_train_time[user])
            user_valid_time[user] = []
            user_valid_time[user].append(User_time[user][-2])
            user_test_time[user] = []
            user_test_time[user].append(User_time[user][-1])
            time_set_test.update(user_test_time[user])


        user_train_valid[user] = user_train[user] + user_valid[user]
        user_train_valid_time[user] = user_train_time[user] + user_valid_time[user]


    # skip = 0
    # neg_f = data_path + args.data + '/' + args.data + '_test_neg.txt'
    # with open(neg_f, 'r') as file:
    #     for line in file:
    #         skip += 1
    #         if skip==1:
    #             continue
    #         user_id, item_id = line.rstrip().split('\t')
    #         u = int(user_id)
    #         i = int(item_id)
    #         usernum = max(u, usernum)
    #         itemnum = max(i, itemnum)

    #         neg_test[u].append(i)

    # sequences = np.zeros((usernum + 1, 101),dtype=np.int64)
    # for user in User:
    #     sequences[user][:] = neg_test[user]

    neg_test = list(np.arange(itemnum+1))
    sequences = np.zeros((usernum + 1, itemnum+1),dtype=np.int64)
    for user in range(1, usernum+1):
        sequences[user][:] = neg_test
        

    neg_test = sequences.copy()
    print(neg_test[1][12101])
    return [user_train, user_valid, user_train_valid, user_test, (user_train_time, user_valid_time, \
        user_train_valid_time, user_test_time, time_set_train, time_set_test), neg_test, itemnum+1, usernum+1]
