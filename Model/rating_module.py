import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import matplotlib
import random
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import copy

PATTERN_LEN = 40
MAX_RATING = 10

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
        # ratings_col = ratings_col.apply(lambda x: min(x, 9))
        self.rating_df_train = rating_df[:3500]
        self.ratings_col_train = ratings_col[:3500]


        ax = self.ratings_col_train.plot.hist(bins=6, alpha=0.5)
        # plt.savefig(f"look.pdf")
        # plt.show()
        # input("continue")
        # plt.clf()

        self.rating_df_train = df_to_tensor(self.rating_df_train)
        self.ratings_col_train = df_to_tensor(self.ratings_col_train, True)

        self.rating_df_test = rating_df[4000:]
        self.ratings_col_test = ratings_col[4000:]
        self.m_factor = 0.9

        self.rating_df_test = df_to_tensor(self.rating_df_test)
        self.ratings_col_test = df_to_tensor(self.ratings_col_test, True)

        self.rating_df_unlabeld = rating_df[3500:4000]
        self.unlabeld_strs = ratings_col[3500:4000]

        self.dropout = nn.Dropout(p=0.1)

        # ax = self.unlabeld_strs.plot.hist(bins=6, alpha=0.5)
        # plt.savefig(f"look.pdf")
        # plt.show()
        # # input("continue")
        # plt.clf()

        self.rating_df_unlabeld = df_to_tensor(self.rating_df_unlabeld)
        self.unlabeld_strs = df_to_tensor(self.unlabeld_strs, True).cpu()

        # self.rating_df_unlabeld = None
        # self.unlabeld_strs = []
        self.unlabeld_events = []
        self.hidden_size1 = 25
        self.hidden_size2 = 15
        self.linear_layer = nn.Linear(PATTERN_LEN, self.hidden_size1).cuda()
        self.linear_layer2 = nn.Linear(self.hidden_size1, self.hidden_size2).cuda()
        self.linear_layer3 = nn.Linear(self.hidden_size2, MAX_RATING).cuda()
        weights = torch.ones(MAX_RATING)
        # weights[0] = 0.2
        # weights[1] = 0.4
        # weights[-1] = 2
        # weights[-2] = 2.5
        # weights[-3] = 1.5
        weights = weights.cuda()
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        train = data_utils.TensorDataset(self.rating_df_train, self.ratings_col_train)
        self.train_loader = data_utils.DataLoader(train, batch_size=64)
        test = data_utils.TensorDataset(self.rating_df_test, self.ratings_col_test)
        self.test_loader = data_utils.DataLoader(test, batch_size=512)
        self.softmax = nn.Softmax(dim=1)
        self.lr = 5e-3
        self.extra_ratings = [[] for _ in range(0, MAX_RATING)]

        self._fix_data_balance(first=True)
        # input("Sdssf")
        # self._create_data_aug(self.rating_df_train[0])

    def label_manually(self, n, weights):
        def del_elements(containter, indexes):
            keep_indexes = list(set(list(range(len(containter)))) - set(indexes))

            if isinstance(containter, list):
                containter = np.array(containter)
                containter = list(containter[keep_indexes])
            else:
                containter = containter[keep_indexes]
            return containter
        if self.rating_df_unlabeld is None:
            raise Exception("do unlabeld data!")
        actuall_size = min(n, len(self.rating_df_unlabeld))
        sampled_indexes = random.choices(range(len(self.rating_df_unlabeld)), k=actuall_size, weights=weights)
        values = self.rating_df_unlabeld[sampled_indexes]
        self.rating_df_train = torch.cat([self.rating_df_train, values])
        user_ratings = []

        # print(f"Len! :{len(self.unlabeld_events)}")
        # input("see")
        if len(self.unlabeld_events) == 0:
            user_ratings = np.array(self.unlabeld_strs)[sampled_indexes]
            # for label in labels:
            #     user_ratings.append(label)

        else:
            for str_pattern, events in zip(np.array(self.unlabeld_strs)[sampled_indexes], np.array(self.unlabeld_events)[sampled_indexes]):
                print(events)
                print(str_pattern)
                user_rating = None
                while user_rating is None:
                    user_rating = input("enter rating ")
                    try:
                        user_rating = int(user_rating)
                    except ValueError:
                        user_rating = int(input("retry: enter rating "))
                user_ratings.append(user_rating)
        user_ratings = torch.tensor(user_ratings).long().cuda()
        self.ratings_col_train = torch.cat([self.ratings_col_train, user_ratings])

        self.rating_df_unlabeld = del_elements(self.rating_df_unlabeld, sampled_indexes)
        self.unlabeld_strs = del_elements(self.unlabeld_strs, sampled_indexes)
        self.unlabeld_events = del_elements(self.unlabeld_events, sampled_indexes)

        train = data_utils.TensorDataset(self.rating_df_train, self.ratings_col_train)
        self.train_loader = data_utils.DataLoader(train, batch_size=40)
        # self.unlabeld_strs = list(self.unlabeld_strs)

    def forward(self, data):
        # data = torch.tensor(data).cuda()
        data = data.cuda()
        data = self.linear_layer(data)
        data = self.dropout(data)
        data = F.relu(data).cuda()
        data = self.linear_layer2(data).cuda()
        data = F.leaky_relu(data).cuda()
        data = self.linear_layer3(data).cuda()
        return data

    def predict(self, data):
        forward_pass = self.forward(data)
        # print(forward_pass.shape)
        # input("softmax shit")
        prediction = self.softmax(forward_pass)
        return prediction

    def get_prediction(self, data, data_str, events, balance_flag=False):
        if not balance_flag:
            if self.rating_df_unlabeld is None:
                self.rating_df_unlabeld = torch.tensor(data)
            else:
                self.rating_df_unlabeld = torch.cat([self.rating_df_unlabeld, data]).clone().detach().requires_grad_(True)
            try:
                self.unlabeld_strs.append(data_str)
            except Exception as e:
                print(type(self.unlabeld_strs))
                print(self.unlabeld_strs)
                raise e

            self.unlabeld_events.append(events)
        res = self.predict(data)
        _, res = torch.max(res, dim=1)
        return res.item() + 1

    def train_single_epoch(self, optimizer, sched, total_count):
        correct = 0
        total_loss = 0
        count_losses, count_all = 0, 0
        distance = 0
        l1_loss = nn.L1Loss()
        self.train()
        for input_x, target in self.train_loader:
            optimizer.zero_grad()
            prediction = self.predict(input_x)
            _, max_val = torch.max(prediction, dim=1)
            loss = self.criterion(prediction, target) * self.m_factor
            # loss = self.criterion(prediction, target)
            # new_distance = torch.mean(abs(max_val - target).float()).requires_grad_(True)
            new_distance = l1_loss(max_val.float(), target.float()).requires_grad_(True)
            loss += new_distance * (1 - self.m_factor)
            distance += new_distance
            # loss = new_distance
            total_loss += loss.item()
            count_losses += 1
            correct += torch.sum(max_val == target).item()
            count_all += len(input_x)
            loss.backward(retain_graph=True)
            optimizer.step()
        if total_count % 10 == 0:
            print(f"Train Avg distance {distance / len(self.train_loader)}")
        sched.step()

        acc = correct / count_all
        return acc

    def test_single_epoch(self, total_count):
        correct = 0
        total_loss = 0
        count_losses, count_all = 0, 0
        all_outputs = None
        distance = 0
        l1_loss = nn.L1Loss()
        self.eval()
        for input_x, target in self.test_loader:
            prediction = self.predict(input_x)
            loss = self.criterion(prediction, target)
            total_loss += loss.item()
            count_losses += 1
            _, max_val = torch.max(prediction, dim=1)
            correct += torch.sum(max_val == target).item()
            # distance += torch.mean(abs(max_val - target).float())
            distance += l1_loss(max_val.float(), target.float()).requires_grad_(True)
            count_all += len(input_x)

        acc = correct / count_all
        if total_count % 10 == 0:
            print(f"Test Avg distance {distance / len(self.test_loader)}")
        if not self.rating_df_unlabeld is None:
            all_outputs = None
            unlabeld = data_utils.TensorDataset(self.rating_df_unlabeld, torch.zeros(len(self.rating_df_unlabeld)))
            unlabeld_loader = data_utils.DataLoader(unlabeld, batch_size=50)
            for input_x, _ in unlabeld_loader:
                prediction = self.predict(input_x)
                if all_outputs is None:
                    all_outputs = prediction
                else:
                    all_outputs = torch.cat((all_outputs, prediction), 0)

        return acc, all_outputs


    def add_pseudo_labels(self, pseudo_labels):
        if self.rating_df_unlabeld is None:
            raise Exception("do unlabeld data!")
        n = 10
        actuall_size = min(n, len(self.rating_df_unlabeld))
        self.rating_df_train = torch.cat([self.rating_df_train, self.rating_df_unlabeld[-actuall_size:]])
        user_ratings = pseudo_labels[-actuall_size:]
        user_ratings = torch.tensor(user_ratings).long().cuda()
        self.ratings_col_train = torch.cat([self.ratings_col_train, user_ratings])
        print(user_ratings)
        print(self.unlabeld_strs[-actuall_size:])
        input("check diff")
        self.rating_df_unlabeld = self.rating_df_unlabeld[:-actuall_size]
        self.unlabeld_strs = self.unlabeld_strs[:-actuall_size]

        # self.unlabeld_events = self.unlabeld_events[actuall_size:]

        train = data_utils.TensorDataset(self.rating_df_train, self.ratings_col_train)
        self.train_loader = data_utils.DataLoader(train, batch_size=40)
        # self.unlabeld_strs = list(self.unlabeld_strs)


    def _train(self, optimizer, sched, count=0, max_count=25, max_total_count=20, n=30):
        torch.allow_unreachable=True
        total_count = 0
        trial_count_reset = 25
        acc, all_outs = self.test_single_epoch(total_count)
        trial_count = trial_count_reset
        acc = 0
        weights = None
        while trial_count > 0:
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != self.lr:
                print(new_lr)
                self.lr = new_lr
                # input("Changed!")
            self.train_single_epoch(optimizer, sched, total_count)
            new_acc, all_outs = self.test_single_epoch(total_count)
            if total_count % 10 == 0:
                print(new_acc)
            if new_acc <= acc:
                trial_count -= 1
            else:
                trial_count = trial_count_reset
                acc = new_acc
            if total_count >= max_total_count:
                print("End!")
                trial_count = 0
            total_count += 1

        if not self.rating_df_unlabeld is None:
            pmax, pmax_indexes = torch.max(all_outs, dim=1)
            pmax = pmax.to("cpu:0")
            weights = torch.tensor([(1 / (val * 2) + 1.5) for val in pmax])
            pmax_indexes = pmax_indexes.to("cpu:0")
            pidx = torch.argsort(pmax)
            pmax_ordered = pmax[pidx]
            pmax_indexes = pmax_indexes[pidx]
            weights = weights[pidx]
            self.rating_df_unlabeld = self.rating_df_unlabeld[pidx]

            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            pidx = pidx.detach().numpy()
            pmax = pmax.detach().numpy()
            self.unlabeld_strs = np.array(self.unlabeld_strs)[pidx.astype(int)].tolist()

            # if count % 5 == 0:
            #     plt.plot(pmax[pidx], color = colors[int(count/5)])
            #     plt.legend([str(i) for i in range(0, count +1, 5)], loc ="lower right")
            #     plt.savefig(f"look.pdf")
            #     plt.show()
            # if new_acc > 0.55:
            #     self.add_pseudo_labels(pmax_indexes)


        if count < max_count:
            self.label_manually(n=n, weights=weights)
            if count % 2 == 0:
                self._fix_data_balance()
            self.m_factor /= 1.1
            self._train(optimizer, sched, count=count+1, max_count=max_count, max_total_count=max_total_count, n=n)


    def _create_data_aug(self, data_inst):
        copy_data = data_inst.clone()
        indexes = random.choices(range(0, len(copy_data)), k=int(PATTERN_LEN / 3))
        for idx in indexes:
            copy_data[idx] = random.choice(copy_data)
        return copy_data


    def _fix_data_balance(self, first=False):

        def _over_sampeling(flatten, split_samples, max_add_extra=50):
            lens_array = np.array([len(i) for i in split_samples])
            print(lens_array)
            mean_len = lens_array.mean()
            for rating, num_exmps in enumerate(lens_array):
                if num_exmps < mean_len * 1.5:
                    if len(self.extra_ratings[rating]) != 0:
                        extras = self.extra_ratings[rating][:max_add_extra]
                        extras = torch.stack(extras)
                        split_samples[rating] = torch.stack(flatten([split_samples[rating], extras]))
                    else:
                        augs = [self._create_data_aug(data_inst) for data_inst in split_samples[rating]]
                        augs = torch.stack(augs)
                        if not first:
                            labels = torch.ones(len(split_samples[rating])) * rating
                            data = split_samples[rating]
                            prediction = self.predict(data)
                            _, max_val = torch.max(prediction, dim=1)
                            max_val = max_val.cpu()
                            falses = max_val != labels
                            augs = augs[falses][:max_add_extra]
                        else:
                            augs = random.choices(augs, k=min(len(augs), max_add_extra))
                        split_samples[rating] = torch.stack(flatten([split_samples[rating], augs]))
                # elif not first and num_exmps < mean_len * 2:
            return split_samples

        def _under_sampeling(split_samples, max_remove=150):
            lens_array = np.array([len(i) for i in split_samples])
            print(lens_array)

            mean_len = lens_array.mean()
            for rating, num_exmps in enumerate(lens_array):
                if num_exmps > mean_len * 2:
                    if not first:
                        labels = torch.ones(len(split_samples[rating])) * rating
                        data = split_samples[rating]
                        prediction = self.predict(data)
                        certain, max_val = torch.max(prediction, dim=1)
                        max_val = max_val.cpu()
                        certain = certain.cpu()
                        to_remove_mask = (max_val == labels) & (certain > 0.55)
                        index = -1
                        while sum(to_remove_mask) > max_remove:
                            to_remove_mask[index] = False
                            index -= 1
                        self.extra_ratings[rating].extend(split_samples[rating][to_remove_mask])
                        split_samples[rating] = split_samples[rating][~to_remove_mask]
                    else:
                        self.extra_ratings[rating].extend(split_samples[rating][:max_remove])
                        split_samples[rating] = split_samples[rating][max_remove:]

            return split_samples


        flatten = lambda list_list: [item for sublist in list_list for item in sublist]
        split_samples = [self.rating_df_train[self.ratings_col_train == i] for i in range(0, MAX_RATING)]
        split_samples = _over_sampeling(flatten, split_samples)
        split_samples = _under_sampeling(split_samples)
        lens_array = np.array([len(i) for i in split_samples])
        print(lens_array)

        self.ratings_col_train = torch.stack(flatten([[torch.tensor(rating)] * len(samples) for rating, samples in enumerate(split_samples)])).cuda()
        split_samples = flatten(split_samples)

        self.rating_df_train = torch.stack(split_samples).cuda()



        self.rating_df_train = self.rating_df_train.cuda()
        train = data_utils.TensorDataset(self.rating_df_train, self.ratings_col_train)
        self.train_loader = data_utils.DataLoader(train, batch_size=64, shuffle=True)


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

            # self.rating_df_unlabeld = torch.tensor(data)
        rating = float(model.pred_pattern.get_prediction(df_to_tensor(predict_pattern), str_pattern, events))
        # print(rating)

        # exit()
    except Exception as e:
        raise e
    rating += 1
    return rating, rating
