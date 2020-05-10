import json


def load_data():
    users = []
    items = []
    ratings = []
    time_weight = []
    cnt = 0

    print("Loading json data...")
    with open(os.path.abspath(opts.data_path + '/train.json'))as f:
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



