# coding: utf-8
import torch
import math
from collections import defaultdict
from collections import OrderedDict


def get_hit_ratio(top_k):
    for k, v in top_k.items():
        if top_k[k][1] == 1:
            return 1
        return 0


def get_ndcg(top_k):
    i = 0
    for k, v in top_k.items():
        i += 1
        if top_k[k][1] == 1:
            return math.log(2) / math.log(i + 2)
        else:
            return 0


def evaluation(evaluation_data, use_cuda, model):
    model.eval()
    hr_mean, ndcg_mean = [], []
    k = [1,5, 10]
    with torch.no_grad():
        t_users = evaluation_data[0]
        t_items = evaluation_data[1]
        t_rating = evaluation_data[2]  # test'rating value is > 0 and negetive sample's rating is 0
        if use_cuda is True:
            t_users, t_items, t_targets = t_users.cuda(), t_items.cuda(), t_rating.cuda()

        predict = model(t_users, t_items)
        full = [t_users.data.view(-1).tolist(),
                t_items.data.view(-1).tolist(),
                predict.data.view(-1).tolist(),
                t_rating.data.view(-1).tolist()]

        user_item_dict = defaultdict(list)
        for i in range(len(full[0])):
            user_item_dict[full[0][i]].append({'item': full[1][i], 'predict': full[2][i], 'rating': full[3][i]})

        for tk in k:
            hr, ndcg = [], []
            for key, v in user_item_dict.items():
                map_item_score = OrderedDict()
                for i in range(len(v)):  # len(v)는 100
                    item = v[i]['item']
                    predict = v[i]['predict']
                    rating = v[i]['rating']
                    map_item_score[item] = (predict, rating)
                top_k = OrderedDict(sorted(map_item_score.items(), key=lambda x: x[1][0], reverse=True)[:tk])
                hr.append(get_hit_ratio(top_k))
                ndcg.append(get_ndcg(top_k))

            hr_mean.append(sum(hr) / len(hr))
            ndcg_mean.append(sum(ndcg) / len(ndcg))
        print('top_k is {}, hit ratio is {}, ndcg is {}'.format(k, hr_mean, ndcg_mean))