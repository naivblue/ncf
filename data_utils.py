import os
import json

def load_data(data_path):
    users = []
    items = []
    ratings = []
    twts = []

    print("Loading json data...")
    with open(os.path.abspath(data_path + '/merge_movie')) as f:
        line = f.readline()
        while line:
            json_data = json.loads(line)
            item = json_data['objectid']
            user = json_data['userid']
            twt = json_data['recency_user']

            #weight = 1.1 / (1.1 - twt) # weighted logistic 모델에서 사용
            users.append(user)
            items.append(item)
            ratings.append(1)
            twts.append(twt)
            line = f.readline()
    print('user_size is {}, item_size is {}, rating_size is {}'.format(len(users),len(items),len(ratings)))
    return users, items, ratings, twts