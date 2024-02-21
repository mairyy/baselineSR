from __future__ import print_function
import sys
import copy
import random
import numpy as np
import pickle
from collections import defaultdict


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    # f = open(fname, 'r')
    with open('data/%s/tst' % fname, 'rb') as out:
        User = pickle.load(out)
    # for line in f:
    #     u, i = line.rstrip().split(' ')
    #     u = int(u)
    #     i = int(i)
    #     usernum = max(u, usernum)
    #     itemnum = max(i, itemnum)
    #     User[u].append(i)

    for user, items in enumerate(User):
        items = [int(item) for item in items]
        nfeedback = len(items)
        maxinlist = max(items)
        itemnum = max(maxinlist, itemnum)
        # nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user+1] = items
            user_valid[user+1] = []
            user_test[user+1] = []
        else:
            user_train[user+1] = items[:-2]
            user_valid[user+1] = []
            user_valid[user+1].append(items[-2])
            user_test[user+1] = []
            user_test[user+1].append(items[-1])
        usernum = user + 1
    print('user %d, item %d', usernum, itemnum)
    return [user_train, user_valid, user_test, usernum, itemnum]

