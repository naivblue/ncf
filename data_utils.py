import os
import json
import collections
from collections import defaultdict
import numpy as np

def load_data():
    users = []
    items = []
    ratings = []
    time_weight = []
    cnt = 0

    print("Loading json data...")
    with open('/data/private/filter_merge_person') as f:
        line = f.readline()
        while line:
            cnt += 1
            if (cnt < 1000001):
                json_data = json.loads(line)
                item = json_data['objectid']
                user = json_data['userid']
                twt = json_data['recency_user']

                weight = 1.1 / (1.1 - twt)
                users.append(user)
                items.append(item)
                ratings.append(1)
                time_weight.append(weight)
                line = f.readline()
            else:
                break
    return users, items, ratings


 # item prob for negative sampling
def item_pro(ratings):
    items = ratings[1]
    item_size = len(items)
    counts = collections.Counter()
    item_p = np.zeros(item_size)

    for item in items:
        counts[item] += 1
    for i in range(item_size):
        item_p[i] = counts[i]
    return item_p/np.sum(item_p)


# item list per user for negative sampling
def user_dict(ratings):
    users = ratings[0]
    items = ratings[1]
    user_dict = defaultdict(list)
    for i in range(len(users)):
        user_dict[users[i]].append(items[i])
    return user_dict