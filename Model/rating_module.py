import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


PATTERN_LEN = 40
MAX_RATING = 11

def df_to_tensor(df, float_flag=False):
    if not float_flag:
        return torch.from_numpy(df.values).float().cuda()
    else:
        return torch.from_numpy(df.values).long().cuda()


class ratingPredictor(nn.Module):
    def __init__(
        self,
        rating_df,
        ratings_col,
    ):
        super().__init__()
        self.rating_df_train = rating_df[:50]
        self.ratings_col_train = ratings_col[:50]

        self.rating_df_train = df_to_tensor(self.rating_df_train)
        self.ratings_col_train = df_to_tensor(self.ratings_col_train, True)

        self.rating_df_test = rating_df[1000:7500]
        self.ratings_col_test = ratings_col[1000:7500]

        self.rating_df_test = df_to_tensor(self.rating_df_test)
        self.ratings_col_test = df_to_tensor(self.ratings_col_test, True)

        self.rating_df_unlabeld = None
        self.unlabeld_strs = []
        self.unlabeld_events = []
        self.linear_layer = nn.Linear(PATTERN_LEN, MAX_RATING).cuda()
        self.criterion = nn.CrossEntropyLoss()
        train = data_utils.TensorDataset(self.rating_df_train, self.ratings_col_train)
        self.train_loader = data_utils.DataLoader(train, batch_size=10)
        test = data_utils.TensorDataset(self.rating_df_test, self.ratings_col_test)
        self.test_loader = data_utils.DataLoader(test, batch_size=50)
        self.softmax = nn.Softmax()

    def label_manually(self, n):
        if self.rating_df_unlabeld is None:
            raise Exception("do unlabeld data!")
        actuall_size = min(n, len(self.rating_df_unlabeld))
        self.rating_df_train = torch.cat([self.rating_df_train, self.rating_df_unlabeld[:actuall_size]])
        user_ratings = []

        for str_pattern, events in zip(self.unlabeld_strs[:actuall_size], self.unlabeld_events[:actuall_size]):
            print(events)
            print(str_pattern)
            user_rating = None
            while user_rating is None:
                user_rating = input("enter rating ")
                try:
                    user_rating = int(user_rating)
                except ValueError:
                    user_rating = input("retry: enter rating ")
            user_ratings.append(user_rating)
        user_ratings = torch.tensor(user_ratings).long().cuda()
        self.ratings_col_train = torch.cat([self.ratings_col_train, user_ratings])

        self.rating_df_unlabeld = self.rating_df_unlabeld[actuall_size:]
        self.unlabeld_strs = self.unlabeld_strs[actuall_size:]
        self.unlabeld_events = self.unlabeld_events[actuall_size:]

        train = data_utils.TensorDataset(self.rating_df_train, self.ratings_col_train)
        self.train_loader = data_utils.DataLoader(train, batch_size=10)

    def forward(self, data):
        data = torch.tensor(data).cuda()
        data = self.linear_layer(data)
        return data

    def predict(self, data):
        forward_pass = self.forward(data)
        prediction = self.softmax(forward_pass)
        return prediction

    def get_prediction(self, data, data_str, events):
        if self.rating_df_unlabeld is None:
            self.rating_df_unlabeld = torch.tensor(data)
        else:
            self.rating_df_unlabeld = torch.cat([self.rating_df_unlabeld, torch.tensor(data)])
        self.unlabeld_strs.append(data_str)
        self.unlabeld_events.append(events)
        res = self.predict(data)
        _, res = torch.max(res, dim=1)
        return res.item()

    def train_single_epoch(self, optimizer):
        correct = 0
        total_loss = 0
        count_losses, count_all = 0, 0

        for input_x, target in self.train_loader:
            optimizer.zero_grad()
            prediction = self.predict(input_x)
            loss = self.criterion(prediction, target)
            total_loss += loss.item()
            count_losses += 1
            _, max_val = torch.max(prediction, dim=1)
            correct += torch.sum(max_val == target).item()

            count_all += len(input_x)
            loss.backward()
            optimizer.step()

        acc = correct / count_all
        return acc

    def test_single_epoch(self):

        correct = 0
        total_loss = 0
        count_losses, count_all = 0, 0
        all_outputs = None
        for input_x, target in self.test_loader:
            prediction = self.predict(input_x)
            loss = self.criterion(prediction, target)
            total_loss += loss.item()
            count_losses += 1
            _, max_val = torch.max(prediction, dim=1)
            correct += torch.sum(max_val == target).item()

            count_all += len(input_x)

        acc = correct / count_all
        print(f"Test Acc: {acc} Loss : {total_loss / count_losses}")

        if not self.rating_df_unlabeld is None:
            unlabeld = data_utils.TensorDataset(self.rating_df_unlabeld, torch.zeros(len(self.rating_df_unlabeld)))
            unlabeld_loader = data_utils.DataLoader(unlabeld, batch_size=50)

            for input_x, _ in unlabeld_loader:
                prediction = self.predict(input_x)
                if all_outputs is None:
                    all_outputs = prediction
                else:
                    all_outputs = torch.cat((all_outputs, prediction), 0)

        return acc, all_outputs

    def train(self, optimizer, count=0, max_count=25):
        acc, all_outs = self.test_single_epoch()
        trial_count = 10
        acc = 0
        while trial_count > 0:
            self.train_single_epoch(optimizer)
            new_acc, all_outs = self.test_single_epoch()
            if new_acc <= acc:
                trial_count -= 1
            else:
                trial_count = 10
                acc = new_acc

        if not self.rating_df_unlabeld is None:
            pmax, _ = torch.max(all_outs, dim=1)
            pmax = pmax.to("cpu:0")
            pidx = torch.argsort(pmax)
            self.rating_df_unlabeld = self.rating_df_unlabeld[pidx]
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            pidx = pidx.detach().numpy()
            pmax = pmax.detach().numpy()
            # self.unlabeld_strs = self.unlabeld_strs[pidx]
            self.unlabeld_strs = np.array(self.unlabeld_strs)[pidx.astype(int)]

            plt.plot(pmax[pidx], color = colors[count])
            plt.legend([str(i) for i in range(0, count + 1)], loc ="lower right")
            plt.savefig(f"look.pdf")
            plt.show()
            self.label_manually(5)


        if count < max_count:
            self.train(optimizer, count=count+1)

def rating_main(model, events, all_conds, actions, str_pattern, rating_flag, pred_flag=False):
    if rating_flag:
        if pred_flag:
            return model_based_rating(model, events, all_conds, str_pattern, actions)
        else:
            return knn_based_rating(model, events, all_conds, str_pattern, actions)
    else:
        return other_rating(model, events, all_conds, actions)


def knn_based_rating(model, events, all_conds, str_pattern, actions):
    flatten = lambda list_list: [item for sublist in list_list for item in sublist]

    predict_pattern = None
    try:
        # for arr_index, arr in enumerate([events, flatten(all_conds), flatten(actions)]):
        for arr_index, arr in enumerate([events, flatten(actions)]):
            arr = arr.copy()
            temp_pd = model.list_of_dfs[arr_index].copy()
            arr += ["Nan"] * (len(temp_pd) - len(arr))
            arr = [temp_pd[array_index][str(val)] for array_index, val in enumerate(arr)]
            to_add = pd.DataFrame(np.array(arr).reshape(-1, len(arr)))

            if predict_pattern is None:
                predict_pattern = to_add
            else:
                predict_pattern = pd.concat([predict_pattern, to_add], axis=1).reset_index(drop=True)
        rating = model.knn.predict(predict_pattern).item()
    except Exception as e:
        print(e)
        # exit()
        rating = model.knn_avg
    if len(events) == 1:
        rating *= 2
    if len(events) >= 3:
        rating += len(events) // 2
    num_eq = str_pattern.count("=")
    rating += num_eq * 0.5
    num_and = str_pattern.count("and")
    num_or = str_pattern.count("or")
    if num_and + num_or > 0:
        if num_and / (num_and + num_or)  < 0.4:
            rating -= 0.2 * num_or
    num_exp = sum(["explosion" in event for event in events]) >= 1
    rating += 0.8 * num_exp

    return rating, (rating + 1.5) - model.knn_avg


def other_rating(model, events, all_conds, actions):
    flatten = lambda list_list: [item for sublist in list_list for item in sublist]

    rating = 1
    if "=" in flatten(all_conds):
        rating *= 1.2
    unique, app_count = np.unique(events, return_counts=True)
    for k in range(len(unique)):
        rating += math.pow(0.7, k + 1) * app_count[k] * 1.3
    if "finish" in events:
        rating += 0.5
    if len(events) > 2 and len(unique) == 1:
        rating *= 0.5
    if len(events) >= 3:
        rating *= len(events) - 1.5
    # rating -= 2
    return rating, rating

def model_based_rating(model, events, all_conds, str_pattern, actions):
    flatten = lambda list_list: [item for sublist in list_list for item in sublist]
    rating = 0
    predict_pattern = None
    try:
        # for arr_index, arr in enumerate([events, flatten(all_conds), flatten(actions)]):
        for arr_index, arr in enumerate([events, flatten(actions)]):
            arr = arr.copy()
            temp_pd = model.list_of_dfs[arr_index].copy()
            arr += ["Nan"] * (len(temp_pd) - len(arr))
            arr = [temp_pd[array_index][str(val)] for array_index, val in enumerate(arr)]
            to_add = pd.DataFrame(np.array(arr).reshape(-1, len(arr)))

            if predict_pattern is None:
                predict_pattern = to_add
            else:
                predict_pattern = pd.concat([predict_pattern, to_add], axis=1).reset_index(drop=True)
        rating = float(model.pred_pattern.get_prediction(df_to_tensor(predict_pattern), str_pattern, events))
        # print(rating)

        # exit()
    except Exception as e:
        raise e
    return rating, rating
