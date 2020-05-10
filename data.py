import torch
import collections
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    def __init__(self, ratings):
        self.user_size = len(set(ratings[0]))
        self.item_size = len(set(ratings[1]))
        self.preprocess_ratings = self._convert_idx(ratings)

        #self.item_p = np.zeros(self.item_size)
        #self.user_dict = defaultdict(list)
        self.item_p = self._item_pro()
        self.user_dict = self._user_dict()
        self.train_ratings, self.test_ratings = self._split_train_test()

    # convert value to idx
    def _convert_idx(self, ratings):
        user = ratings[0]
        item = ratings[1]
        rating = ratings[2]

        user_to_ix = {value: i for i, value in enumerate(set(user))}
        item_to_ix = {value: i for i, value in enumerate(set(item))}
        user_idxs = [user_to_ix[i] for i in user]
        item_idxs = [item_to_ix[i] for i in item]
        return user_idxs, item_idxs, rating

    # item prob for negative sampling
    def _item_pro(self):
        items = self.preprocess_rating[1]
        counts = collections.Counter()
        item_p = np.zeros(self.item_size)

        for item in items:
            counts[item] += 1
        for i in range(self.item_size):
            print(i,counts[i])
            item_p[i] = counts[i]
        item_p /= np.sum(item_p)
        print(item_p)
        return item_p

    # item list per user for negative sampling
    def _user_dict(self):
        users = self.preprocess_ratings[0]
        items = self.preprocess_ratings[1]
        user_dict = defaultdict(list)
        for i in range(len(users)):
            user_dict[users[i]].append(items[i])
        return user_dict

    # split train test data
    def _split_train_test(self):
        train_users = []
        train_items = []
        train_ratings = []
        test_users = []
        test_items = []
        test_ratings = []
        test_cnt = 0

        users_idx = self.preprocess_ratings[0]
        items_idx = self.preprocess_ratings[1]

        train_user = defaultdict(int)
        for i, user in enumerate(users_idx):
            train_user[user] += 1

            # test data는 user-item pair가 2개 이상인 user만 대상으로 함
            # test data에 있는 user는 train에 반드시 포함되어 있는 user이며, test는 user당 1개의 user-item pair데이터만 가짐
            if train_user[user] == 2 and test_cnt < 10000:
                test_cnt += 1
                test_users.append(users_idx[i])
                test_items.append(items_idx[i])
                test_ratings.append(self.preprocess_ratings[2][i])
            else:
                train_users.append(users_idx[i])
                train_items.append(items_idx[i])
                train_ratings.append(self.preprocess_ratings[2][i])

        return (train_users, train_items, train_ratings), (test_users, test_items, test_ratings)


    # negative sampling batch train data
    def train_data_loader(self, ng_sample_size, batch_size):
        users = self.train_ratings[0]
        items = self.train_ratings[1]
        rating = self.train_ratings[2]

        uniq_users = set(users)
        # negative sampling
        for u in uniq_users:
            iters = 0
            while iters < ng_sample_size:
                j = np.random.choice(self.item_size, p=self.item_p)
                if j not in self.user_dict[u]:
                    users.append(u)
                    items.append(j)
                    rating.append(0)
                    iters += 1
                else:
                    iters += 0

        user_tensor = torch.LongTensor(users)
        item_tensor = torch.LongTensor(items)
        rating_tensor = torch.FloatTensor(rating)
        train_dataset = TensorDataset(user_tensor, item_tensor, rating_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader



    # negative sampling test data
    def test_data_loader(self, ng_sample_size):
        users = self.test_ratings[0]
        items = self.test_ratings[1]
        rating = self.test_ratings[2]

        uniq_users = set(users)
        # negative sampling
        for u in uniq_users:
            iters = 0
            while iters < ng_sample_size:
                j = np.random.choice(self.item_size, p=self.item_p)
                if j not in self.user_dict[u]:
                    users.append(u)
                    items.append(j)
                    rating.append(0)
                    iters += 1
                else:
                    iters += 0

        user_tensor = torch.LongTensor(users)
        item_tensor = torch.LongTensor(items)
        rating_tensor = torch.FloatTensor(rating)
        test_loader = [user_tensor, item_tensor, rating_tensor]
        return test_loader