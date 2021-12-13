import argparse
import torch
import random
import torch.nn as nn
import os
from copy import deepcopy

from Model.utils import (
    OpenCEP_pattern,
    after_epoch_test,
    new_mapping,
    get_action_type,
    create_pattern_str,
    ball_patterns,
    bayesian_function,
    set_values_bayesian,
    store_to_file,
    replace_values,
    run_OpenCEP,
)
import tqdm
import pathlib
from bayes_opt import BayesianOptimization
import ast
import sys
import time
from itertools import count
from multiprocessing import Process, Queue
import torch.nn.functional as F
from torch.autograd import Variable
import torch.cuda as cuda_handle
import gc
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from shutil import copyfile
import datetime
from difflib import SequenceMatcher
import pandas as pd
from stream.FileStream import FileInputStream, FileOutputStream
from sklearn.neighbors import KNeighborsClassifier as KNN
import wandb
from bayes_opt import BayesianOptimization

GRAPH_VALUE = 50
GAMMA = 0.99
EMBDEDDING_TOTAL_SIZE = 21
PAD_VALUE = -5.5
class_inst = None

class ruleMiningClass(nn.Module):
    def __init__(
        self,
        data_path,
        pattern_path,
        events_path,
        num_events,
        match_max_size=8,
        max_values=None,
        normailze_values=None,
        window_size=350,
        max_fine_app=55,
        eff_cols=None,
        all_cols=None,
        max_time=0,
        lr=1e-6,
        init_flag=False,
        hidden_size1=512,
        hidden_size2=2048,
    ):
        super().__init__()
        self.lr = lr
        # self.actions = [">", "<", "=", "+>", "->", "*="]
        self.actions = [">", "<", "="]
        # self.actions = [">", "<", "=", "+>", "->", "*]
        self.max_predict = (match_max_size + 1) * (len(eff_cols) + 1)
        self.events = np.loadtxt(events_path, dtype='int')
        self.num_events = len(self.events)
        self.match_max_size = match_max_size
        self.max_values = max_values
        self.window_size = window_size
        self.normailze_values = normailze_values
        self.embedding_events = nn.Embedding(num_events + 1, 3)
        self.embedding_values = [nn.Embedding(max_val, 3) for max_val in max_values]
        if init_flag:
            if not os.path.exists(f"Processed_Data/{self.window_size}.pt"):
                self.data = self._create_data(data_path)
                self.data = self.data.view(len(self.data), -1)
                self.data = self.data.detach().clone().requires_grad_(True)
                torch.save(self.data, f"Processed_Data/{self.window_size}.pt")
            else:
                self.data = torch.load(f"Processed_Data/{self.window_size}.pt").requires_grad_(True)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_cols = len(eff_cols)
        self.num_actions = (len(self.actions) * (self.match_max_size + 1)) * 2 * 2 + 1  # [>|<|= * [match_max_size + 1(value)] * not / reg] * (and/or) |nop
        self.embedding_actions = nn.Embedding(self.num_actions + 1, 1)
        self.embedding_desicions = nn.Embedding(self.num_events + 1, 1)
        # self.linear_base = nn.Linear(
        #     self.window_size * EMBDEDDING_TOTAL_SIZE + (self.match_max_size + 1)  * (self.num_cols + 1),
        #     self.hidden_size2,
        # ).cuda()
        self.linear_base = nn.Linear(
            self.window_size * EMBDEDDING_TOTAL_SIZE,
            self.hidden_size1,
        ).cuda()
        self.dropout = nn.Dropout(p=0.2).cuda()
        self.linear_finish = nn.Linear(
            self.hidden_size1 + (self.match_max_size + 1) * (self.num_cols + 1),
            self.hidden_size2,
        ).cuda()
        self.event_tagger = nn.Linear(self.hidden_size2, self.num_events + 1).cuda()

        self.action_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size2, self.num_actions)
                for _ in range(self.num_cols)
            ]
        ).cuda()
        self.critic_reward = nn.Linear(self.hidden_size2, 1).cuda()
        self.critic_rating = nn.Linear(self.hidden_size2, 1).cuda()

        self._create_training_dir(data_path)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.all_cols = all_cols
        self.cols = eff_cols
        self.max_fine_app = max_fine_app
        self.knn_avg = 0
        if not pattern_path == "":
            self.knn = self._create_df(pattern_path)
        self.max_time = max_time
        self.count = 0
        self.min_values_bayes = [-i for i in normailze_values]
        self.max_values_bayes = [i - j for i,j in zip(max_values, normailze_values)]


    def layers_based_hidden(self, hidden_size):
        self.hidden_size1 = hidden_size
        self.linear_base = nn.Linear(
            self.window_size * EMBDEDDING_TOTAL_SIZE,
            self.hidden_size1,
        ).cuda()
        self.linear_finish = nn.Linear(
            self.hidden_size1 + (self.match_max_size + 1) * (self.num_cols + 1),
            self.hidden_size2,
        ).cuda()

    def layers_based_hidden2(self, hidden_size):
        self.linear_finish = nn.Linear(
            self.hidden_size1 + (self.match_max_size + 1) * (self.num_cols + 1),
            self.hidden_size2,
        ).cuda()
        self.event_tagger = nn.Linear(self.hidden_size2, self.num_events + 1).cuda()

        self.action_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size2, self.num_actions)
                for _ in range(self.num_cols)
            ]
        ).cuda()
        self.critic_reward = nn.Linear(self.hidden_size2, 1).cuda()
        self.critic_rating = nn.Linear(self.hidden_size2, 1).cuda()

    def layers_hidden(self, hidden_size1, hidden_size2):
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.linear_base = nn.Linear(
            self.window_size * EMBDEDDING_TOTAL_SIZE,
            self.hidden_size1,
        ).cuda()
        self.linear_finish = nn.Linear(
            self.hidden_size1 + (self.match_max_size + 1) * (self.num_cols + 1),
            self.hidden_size2,
        ).cuda()
        self.event_tagger = nn.Linear(self.hidden_size2, self.num_events + 1).cuda()

        self.action_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size2, self.num_actions)
                for _ in range(self.num_cols)
            ]
        ).cuda()
        self.critic_reward = nn.Linear(self.hidden_size2, 1).cuda()
        self.critic_rating = nn.Linear(self.hidden_size2, 1).cuda()


    def _create_df(self, pattern_path):
        def fix_str_list_columns_init(data, flag=False):
            data = data[1:]
            data = data[:-1]
            data = data.replace("\"", "")
            data = data.replace("\'", "")
            data = data.replace(" ", "")
            temp = pd.Series(data)
            temp = temp.str.split(",", expand=True)
            return temp


        def fix_str_list_columns(temp):
            for col in temp.columns:
                temp[col] = temp[col].astype('category')
                temp[col] = temp[col].cat.codes
            return temp


        self.list_of_dfs = []
        df = pd.read_csv(pattern_path)[["rating", "events", "conds", "actions"]]

        df.rating = df.rating.apply(lambda x : round(float(x) + 0.5))
        str_list_columns = ["conds", "actions"]
        int_list_columns = ["events"]
        fit_columns = int_list_columns + str_list_columns
        df_new = None
        for col in fit_columns:
            temp = None

            for val in df[col]:
                if temp is None:
                    temp = fix_str_list_columns_init(val)
                else:
                    temp = temp.append(fix_str_list_columns_init(val))
            temp = temp.reset_index(drop=True)

            add_df = []
            for col_name in temp.columns:
                temp_dict = dict(zip(temp[col_name],temp[col_name].astype('category').cat.codes))
                temp_dict['Nan'] = -1
                add_df.append(temp_dict)
            self.list_of_dfs.append(add_df)
            combined = fix_str_list_columns(temp)
            combined.columns = list(map(lambda x: col + "_" + str(x), combined.columns))

            if df_new is None:
                df_new = combined
            else:
                df_new = pd.concat([df_new, combined], axis=1).reset_index(drop=True)
            df_new = df_new.fillna(PAD_VALUE)
        knn = KNN(n_neighbors=1)
        knn.fit(df_new, df["rating"])
        self.knn_avg = df.rating.mean()
        return knn

    def _create_data(self, data_path):
        date_time_obj = None
        data = None
        print(data_path)
        with open(data_path) as f:
            for line in f:
                values = line.split("\n")[0]
                values = values.split(",")
                event = values[0]
                event = self.embedding_events(torch.tensor(int(new_mapping(event, self.events, reverse=True))))
                values = values[2:] # skip sid and ts
                try:
                    embed_values = [self.embedding_values[i](torch.tensor(int(value) + self.normailze_values[i])) for (i,value) in enumerate(values[:len(self.normailze_values)])]
                    embed_values.insert(0, event)
                except Exception as e:
                    embed_values = []
                    print(e)
                    print(values)
                    print(len(values))
                    print(line)
                    for i, value in enumerate(values[:len(self.normailze_values)]):
                        print(i)
                        print(value)
                        a = self.normailze_values[i]
                        print(a)
                        a = torch.tensor(int(value) + a)
                        print(a)
                        a = self.embedding_values[i](a)
                        print(a)
                        embed_values.append(a)
                if data is None:
                    data = torch.cat(tuple(embed_values), 0)
                    # filler = torch.tensor([PAD_VALUE] * (self.max_predict - len(data)))
                    # data = torch.cat((data, filler))
                    data = data.unsqueeze(0)
                else:
                    new_data = torch.cat(tuple(embed_values), 0)
                    # filler = torch.tensor([PAD_VALUE] * (self.max_predict - len(new_data)))
                    # new_data = torch.cat((new_data, filler))

                    new_data = new_data.unsqueeze(0)
                    data = torch.cat((data, new_data), 0)

        sliding_window_data = None
        for i in range(0, len(data) - self.window_size):
            if i % 1000 == 0:
                print(i)
            if sliding_window_data is None:
                sliding_window_data = data[i : i + self.window_size]
                sliding_window_data = sliding_window_data.unsqueeze(0)
            else:
                to_add = data[i : i + self.window_size].unsqueeze(0)
                sliding_window_data = torch.cat((sliding_window_data, to_add))
        return sliding_window_data

    def _create_training_dir(self, data_path):
        if not os.path.exists("Model/training/"):
            os.mkdir("Model/training/")
        lines = []
        with open(data_path) as f:
            for line in f:
                lines.append(line)

        for i in range(0, len(lines) - self.window_size):
            with open("Model/training/{}.txt".format(i), "w") as f:
                for j in range(i, i + self.window_size):
                    f.write(lines[j])

    def forward(self, input, old_desicions, mask, T=1):
        def masked_softmax(vec, mask, dim=1, T=1):
            vec = vec / T
            masked_vec = vec.cpu() * mask.float()
            max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
            exps = torch.exp(masked_vec - max_vec).cpu()
            masked_exps = exps * mask.float()
            masked_sums = masked_exps.sum(dim, keepdim=True)
            zeros = masked_sums == 0
            masked_sums += zeros.float()
            return masked_exps / masked_sums

        x = self.dropout(self.linear_base(input.cuda())).cuda()
        combined = torch.cat((x, old_desicions.detach().clone().requires_grad_(True).cuda())).cuda()
        attention_mask = torch.tensor(list(np.random.uniform(0.1, 0.5, len(x))) + [1] * len(old_desicions)).cuda()
        combined *= attention_mask

        after_relu = F.relu(self.linear_finish(combined))
        # x = F.relu(self.linear_base(input.cuda()))

        value_reward = self.critic_reward(after_relu)
        value_rating = self.critic_rating(after_relu)
        event_before_softmax = self.event_tagger(after_relu)
        event_after_softmax = masked_softmax(event_before_softmax, mask.clone(), dim=0, T=T)
        return event_after_softmax, value_reward, value_rating

    def get_event(self, input,old_desicions, index=0, mask=None, training_factor=0.0, T=1):
        # print(input.shape)
        probs, value_reward, value_rating = self.forward(Variable(input), Variable(old_desicions), mask, T=T)
        numpy_probs = probs.detach().cpu().numpy()
        action = np.random.choice(
            self.num_events + 1, p=np.squeeze(numpy_probs)
        )
        entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2
        if np.random.rand() > 1 - training_factor:
            action = np.random.randint(len(mask))
        if index % 50 == 0:
            print(probs)
            # print(log_prob)
            # sleep(1)

        log_prob = torch.log(probs.squeeze(0)[action])
        if abs(log_prob) < 0.1:
            self.count += 1

        return action, log_prob, value_reward, value_rating, entropy


    def single_col_mini_action(self, data, index, training_factor=0.0):
        def masked_softmax(vec, mask, dim=0, T=1):
            vec = vec / T
            masked_vec = vec.cpu() * mask.float()
            max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
            exps = torch.exp(masked_vec - max_vec).cpu()
            masked_exps = exps * mask.float()
            masked_sums = masked_exps.sum(dim, keepdim=True)
            zeros = masked_sums == 0
            masked_sums += zeros.float()
            return masked_exps / masked_sums


        x = F.relu(self.action_layers[index](data))
        mask = [1.0] * self.num_actions
        mask[-1] = 20
        mask = torch.tensor([float(i)/sum(mask) for i in mask])

        probs = masked_softmax(x, mask=mask, dim=0)
        numpy_probs = probs.detach().cpu().numpy()
        entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2
        if np.random.rand() > 1 - training_factor:
            highest_prob_action = np.random.randint(len(probs))
        else:
            highest_prob_action = np.random.choice(
                self.num_actions, p=np.squeeze(numpy_probs)
            )
        # print(numpy_probs)
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action]).cpu()
        highest_prob_value = None
        mini_action, _, _ = get_action_type(
            highest_prob_action, self.num_actions, self.actions, self.match_max_size
        )

        if len(mini_action.split("value")) > 1:
            highest_prob_value = "value"

        return highest_prob_action, highest_prob_value, log_prob, entropy

    def get_cols_mini_actions(self, data, old_desicions, training_factor=0.0):
        mini_actions = []
        log_probs = 0.0
        compl_vals = []
        conds = []
        mini_actions_vals = []
        total_entropy = 0
        comps_to = []
        after_base = self.dropout(self.linear_base(data))
        updated_data = torch.cat((after_base, old_desicions.detach().clone().requires_grad_(True).cuda()))
        attention_mask = torch.tensor(list(np.random.uniform(0.1, 0.5, len(after_base))) + [1] * len(old_desicions)).cuda()
        updated_data *= attention_mask

        after_relu = F.relu(self.linear_finish(updated_data))
        # updated_data = self.linear_base(data)
        for i in range(self.num_cols):
            action, value, log, entropy = self.single_col_mini_action(after_relu, i, training_factor) #this is weird, should update data after actions
            mini_actions_vals.append(action)
            total_entropy += entropy / self.num_cols
            mini_action, cond, comp_to = get_action_type(action, self.num_actions, self.actions, self.match_max_size)
            conds.append(cond)
            comps_to.append(comp_to)
            if len(mini_action.split("value")) > 1:
                mini_action = mini_action.replace("value", "")  # TODO:replace this shit
                compl_vals.append(value)  # TODO: change
            else:
                compl_vals.append("nop")  # TODO: change
            mini_actions.append(mini_action)
            log_probs += log / self.num_cols
        return mini_actions, log_probs, compl_vals, conds, mini_actions_vals, total_entropy, comps_to


def update_policy1(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std(unbiased=False) + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward(retain_graph=True)
    policy_network.optimizer.step()


def update_policy(policy_network, rewards, log_probs, values, Qval, entropy_term, flag=False):
    Qvals = np.zeros_like(values)
    for t in reversed(range(len(rewards))):
        Qval = rewards[t] + GAMMA * Qval
        Qvals[t] = Qval

    values = torch.FloatTensor(values)
    Qvals = torch.FloatTensor(Qvals)
    log_probs = torch.stack(log_probs)
    advantage = Qvals - values

    policy_loss = [-log_prob * adv for (log_prob, adv) in zip(log_probs, advantage)]
    value_loss = torch.tensor([F.smooth_l1_loss(torch.tensor([value]), torch.tensor([Qval])) for (value, Qval) in zip(values, Qvals)])

    policy_network.optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + value_loss.sum()
    loss = loss.cuda()
    loss.backward(retain_graph=True)
    policy_network.optimizer.step()
    # actor_loss = (-log_probs * advantage).mean()
    # critic_loss = 0.5 * advantage.pow(2).mean()
    # policy_gradient = actor_loss + critic_loss + 0.001 * entropy_term
    # policy_gradient = policy_gradient.cuda()
    # policy_network.optimizer.zero_grad()
    # policy_gradient.backward(retain_graph=True)
    # policy_network.optimizer.step()


def train(model, num_epochs=5, test_epcohs=False, split_factor=0, bs=0, rating_flag=True):
    # run_name = "split=" + str(int(split_factor * 100)) + "_window_size=" + str(model.window_size)
    # run_name = "hidden2=" + str(model.hidden_size2) + "_window_size=" + str(model.window_size)
    # run_name = "copy:hidden1=" + str(model.hidden_size1) + "_hidden2=" + str(model.hidden_size2) + "_window_size=" + str(model.window_size)
    run_name = "test_run5"
    not_finished_count = 0
    # run_name = "second_level_" + str(model.window_size)
    run = wandb.init(project='Pattern_Mining', entity='guyshapira', name=run_name, settings=wandb.Settings(start_method='fork'))
    # wandb.watch(model)
    config = wandb.config
    config.hidden_size1 = model.hidden_size1
    config.hidden_size2 = model.hidden_size2
    config.learning_rate = model.lr
    config.batch_size = bs
    config.window_size = model.window_size
    config.num_epochs = num_epochs
    config.split_factor = split_factor
    flatten = lambda list_list: [item for sublist in list_list for item in sublist]
    torch.autograd.set_detect_anomaly(True)
    added_info_size = (model.match_max_size + 1) * (model.num_cols + 1)
    # added_info_size = model.max_predict
    total_best = -1
    best_found = {}
    results, all_rewards, numsteps, avg_numsteps, mean_rewards, real, mean_real, rating_plot, all_ratings, factor_results = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    max_rating = []
    entropy_term, turn_flag = 0, 0
    training_factor = 0.8
    switch_flag = int(split_factor * bs)
    pbar_file = sys.stdout
    total_count = -25
    # total_count = 0

    for epoch in range(num_epochs):
        print(f"Not finished = {not_finished_count}\n")
        if epoch >= 1:
            for g in model.optimizer.param_groups:
                g['lr'] *= 0.8

        if epoch < 2:
            temper = 0.25
        else:
            temper = 1

        with tqdm.tqdm(total=bs, file=pbar_file) as pbar:
            in_round_count = 0
            for index in range(epoch, len(model.data), len(model.data) // bs):
                if total_count >= bs:
                    total_count = 0
                    turn_flag = 1 - turn_flag

                if total_count == switch_flag:
                    turn_flag = 1 - turn_flag
                if in_round_count >= bs:
                    break
                total_count += 1
                in_round_count += 1
                data = model.data[index]
                data_size = len(data)
                old_desicions = torch.tensor([PAD_VALUE] * added_info_size)
                # old_desicions = old_desicions.unsqueeze(0)
                # data = torch.cat((data, old_desicions.float()))
                count = 0
                best_reward = 0.0
                pbar.update(n=1)
                is_done = False
                events = []
                actions, rewards, log_probs, action_types, real_rewards, all_conds, all_comps = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                values, comp_values, patterns, ratings = (
                    [],
                    [],
                    [],
                    [],
                )
                normalize_rating, normalize_reward = [], []
                mask = torch.tensor([1.0] * (model.num_events + 1))
                if in_round_count % 35 == 0 and epoch < 5:
                    temper /= 1.05
                if total_count % 25 == 0:
                    training_factor /= 1.2
                while not is_done:
                    data = data.cuda()
                    mask_orig = mask.clone()
                    action, log_prob, value_reward, value_rating, entropy = model.get_event(
                        data, old_desicions, in_round_count, mask, training_factor=training_factor, T=temper
                    )
                    old_desicions = old_desicions.clone()
                    old_desicions[count * (model.num_cols + 1)] = model.embedding_desicions(
                        torch.tensor(action)
                    ).cuda()
                    count += 1
                    if turn_flag == 0 and False:
                        value = value_rating
                    else:
                        value = value_reward
                    value = value.detach().cpu().numpy()[0]
                    values.append(value)
                    entropy_term += entropy

                    if action == model.num_events:
                        ratings.append(1)
                        if len(actions) == 0:
                            log_probs.append(log_prob)
                            rewards.append(-1.5)
                            real_rewards.append(-1.5)
                        else:
                            log_probs.append(log_prob)
                            rewards.append(10)
                            real_rewards.append(10)
                            break
                    else:
                        mask[-1] *= 1.1
                        mask[action] *= 1.25
                        event = new_mapping(action, model.events)
                        events.append(event)
                        mini_actions, log, comp_vals, conds, actions_vals, entropy, comps_to = \
                            model.get_cols_mini_actions(data, old_desicions, training_factor=training_factor)
                        all_comps.append(comps_to)
                        entropy_term += entropy
                        for j, action_val in enumerate(actions_vals):
                            old_desicions = old_desicions.clone()
                            old_desicions[count * (model.num_cols + 1) + j + 1] = model.embedding_actions(torch.tensor(action_val))
                        log_prob += log.item()
                        log_probs.append(log_prob)
                        actions.append(mini_actions)


                        currentPath = pathlib.Path(os.path.dirname(__file__))
                        absolutePath = str(currentPath.parent)
                        sys.path.append(absolutePath)
                        file = os.path.join(absolutePath, "Model", "training", "{}.txt".format(index))

                        all_conds.append(conds)


                        if comp_vals.count("nop") != len(comp_vals):
                            bayesian_dict = set_values_bayesian(comp_vals,
                                model.all_cols, model.cols, mini_actions, event,
                                all_conds, file, model.max_values_bayes,
                                model.min_values_bayes
                             )
                            store_to_file(events, actions, index, comp_values, model.cols, all_conds, comp_vals, all_comps, model.max_fine_app, model.max_time)
                            b_optimizer = BayesianOptimization(
                                f=bayesian_function,
                                pbounds=bayesian_dict,
                                random_state=42,
                                verbose=0,
                            )
                            try:
                                b_optimizer.maximize(
                                    init_points=2,
                                    n_iter=3,
                                )
                                selected_values = list(b_optimizer.max['params'].values())
                            except Exception as e:
                                # empty range, just use min to max values as range instade
                                selected_values = [max(model.normailze_values) for _ in range(len(bayesian_dict))]
                            comp_vals = replace_values(comp_vals, selected_values)

                        comp_values.append(comp_vals)
                        finished_flag = True
                        try:
                            pattern = OpenCEP_pattern(
                                events, actions, index, comp_values, model.cols, all_conds, all_comps, model.max_time
                            )
                        except Exception as e:
                            # timeout error
                            finished_flag = False
                            not_finished_count += 1

                        str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols, all_comps)
                        if rating_flag:
                            predict_pattern = None
                            try:
                                for arr_index, arr in enumerate([events, flatten(all_conds), flatten(actions)]):
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
                                rating = model.knn_avg
                            if len(events) >= 3:
                                rating *= 1.25
                            if "=" in flatten(all_conds):
                                rating *= 1.2
                            ratings.append(rating)
                            rating -= (model.knn_avg - 0.5)
                            normalize_rating.append(rating)
                        else:
                            rating = 1
                            ratings.append(rating)
                            normalize_rating.append(rating)

                        if not finished_flag:
                            reward = -5
                            rewards.append(reward)
                            real_rewards.append(reward)
                            normalize_reward.append(reward - 20)
                            is_done = True
                        else:
                            eff_pattern = pattern.condition

                            with open("Data/Matches/{}Matches.txt".format(index), "r") as f:
                                content = f.read()
                                reward = int(content.count("\n") / (len(actions) + 1))
                                if reward > model.max_fine_app:
                                    reward = -1
                                reward *= (1 + (len(events) - 1)/5)
                                real_rewards.append(reward)
                                if reward == 0:
                                    normalize_reward.append(-25)
                                    rewards.append(-1.5)
                                    break
                                normalize_reward.append(reward - 20)
                                reward *= rating
                                rewards.append(reward)
                                if len(best_found) < 10:
                                    best_found.update({reward: pattern})
                                else:
                                    worst_reward = sorted(list(best_found.keys()))[0]
                                    if reward > worst_reward:
                                        del best_found[worst_reward]
                                        best_found.update({reward: pattern})


                    if count >= model.match_max_size:
                        is_done = True

                _, Qval_reward, Qval_rating = model.forward(data, torch.tensor([PAD_VALUE] * added_info_size), mask, T=temper)
                if turn_flag == 0 and False:

                    Qval = Qval_rating.detach().cpu().numpy()[0]
                else:
                    Qval = Qval_reward.detach().cpu().numpy()[0]
                del data
                gc.collect()
                if total_count < 0 or True:
                    send_rewards = rewards
                elif turn_flag == 0:
                    send_rewards = ratings
                else:
                    send_rewards = real_rewards
                if total_count % bs == 0:
                    update_policy(model, send_rewards, log_probs, values, Qval, entropy_term, flag=True)
                else:
                    update_policy(model, send_rewards, log_probs, values, Qval, entropy_term)


                index_max = np.argmax(rewards)
                all_ratings.append(np.sum(ratings))
                all_rewards.append(rewards[index_max])
                numsteps.append(len(actions))
                avg_numsteps.append(np.mean(numsteps))
                mean_rewards.append(np.mean(all_rewards))
                max_rating.append(np.max(ratings))
                real.append(real_rewards[index_max])
                rating_plot.append(ratings[index_max])
                mean_real.append(np.mean(real_rewards))

                if in_round_count % 2 == 0:
                    sys.stdout.write(
                        "\nReal reward : {}, Rating {},  comparisons : {}\n".format(
                            real_rewards[index_max],
                            ratings[index_max],
                            sum([t != "nop" for sub in comp_values for t in sub]),
                        )
                    )
                    str_pattern = create_pattern_str(events[:index_max + 1], actions[:index_max + 1],
                     comp_values[:index_max + 1], all_conds[:index_max + 1], model.cols, all_comps[:index_max + 1])
                    sys.stdout.write(f"Pattern: events = {events[:index_max + 1]}, conditions = {str_pattern} index = {index}\n")
                    sys.stdout.write(
                        "episode: {}, index: {}, total reward: {}, average_reward: {}, length: {}\n".format(
                            in_round_count,
                            index,
                            np.round(rewards[index_max], decimals=3),
                            np.round(np.mean(all_rewards), decimals=3),
                            index_max + 1,
                        )
                    )
                if model.count > 15:
                    print("\n\n\n---- Stopping early because of low log ----\n\n\n")
                    model.count = 0
                    for g in model.optimizer.param_groups:
                        g['lr'] *= 0.5
                    break


            rating_groups = [
                np.mean(rating_plot[t : t + GRAPH_VALUE])
                for t in range(0, len(rating_plot), GRAPH_VALUE)
            ]
            max_ratings_group = [
                np.mean(max_rating[t: t + GRAPH_VALUE])
                for t in range(0, len(max_rating), GRAPH_VALUE)
            ]
            real_groups = [
                np.mean(real[t : t + GRAPH_VALUE])
                for t in range(0, len(real), GRAPH_VALUE)
            ]
            for rew, rat, max_rat in zip(real_groups[-int(bs / GRAPH_VALUE):], rating_groups[-int(bs / GRAPH_VALUE):], max_ratings_group[-int(bs / GRAPH_VALUE):]):
                wandb.log({"reward_test": rew, "rating_test": rat, "max rating_test": max_rat, 'lr':model.lr})
            fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)

            ax1.set_xlabel("Episode")
            ax1.set_title("Reward vs number of episodes played")
            labels = [
                "{}-{}".format(t, t + GRAPH_VALUE)
                for t in range(0, len(real), GRAPH_VALUE)
            ]
            locations = [
                t + int(GRAPH_VALUE / 2) for t in range(0, len(real), GRAPH_VALUE)
            ]
            plt.sca(ax1)
            plt.xticks(locations, labels)

            ax1.scatter(locations, real_groups, c="g")
            ax1.set_ylabel("Avg Matches per window")

            ax1.plot()

            locations = [
                t + int(GRAPH_VALUE / 2) for t in range(0, len(rating_plot), GRAPH_VALUE)
            ]
            ax2.set_ylabel("Avg Rating per window")
            ax2.set_xlabel("Episode")
            ax2.set_title("Rating vs number of episodes played")
            plt.sca(ax2)
            plt.xticks(locations, labels)

            ax2.scatter(locations, rating_groups, c="g")
            ax2.scatter(locations, max_ratings_group, c="r")
            ax2.plot()
            str_split_factor = str(split_factor * 100) + "%"
            if not os.path.exists(f"Graphs/{str_split_factor}/"):
                os.mkdir(f"Graphs/{str_split_factor}/")
            plt.savefig(f"Graphs/{str_split_factor}/{str(len(real))}_{model.window_size}.pdf")
            plt.show()
            factor_results.append({"rating" : rating_groups[-1], "reward": real_groups[-1]})


            if False:
                after_epoch_test(best_pattern)
                with open("Data/Matches/allMatches.txt", "r") as f:
                    results.append(int(f.read().count("\n") / (max_len_best + 1)))
                os.remove("Data/Matches/allMatches.txt")
    if test_epcohs:
        print(results)
        plt.plot(results, "g")
        plt.show()

    run.finish()
    cuda_handle.empty_cache()
    best_res = - 10
    for dict_res in factor_results:
        new_res = dict_res['rating'] / 10 + dict_res['reward'] / model.max_fine_app
        if new_res > best_res:
            best_res = new_res
    return best_res
    # return factor_results[-1], best_found


# def predict_window(model, i, data):
#     data_size = len(data)
#     added_info_size = (model.match_max_size + 1) * (model.num_cols + 1)
#     added_info_size_knn = (model.match_max_size + 1) * (6 + 1)
#     old_desicions = torch.tensor([0] * added_info_size)
#     data2 = torch.cat((data, torch.tensor([0] * added_info_size_knn)), dim=0)
#     data = torch.cat((data, old_desicions.float()), dim=0)
#     count = 0
#     is_done = False
#     events = []
#     actions, rewards, action_types, all_conds, comp_values, ratings, all_comps = (
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#     )
#     str_pattern = ""
#     mask = torch.tensor([1.0] * (model.num_events + 1))
#     pattern = None
#     while not is_done:
#         action, _, _, _ = model.get_event(data, i, mask.detach())
#         data = data.clone()
#         data[data_size + count * (model.num_cols + 1)] = model.embedding_desicions(
#             torch.tensor(action)
#         ).cuda()
#         data2[data_size + count * (model.num_cols + 1)] = data[data_size + count * (model.num_cols + 1)]
#         count += 1
#         if action == model.num_events:
#             # mask[-1] = mask[-1].clone() * 1.1
#             if len(actions) != 0:
#                 ratings = ratings[:-1]
#                 data[data_size + count * (model.num_cols + 1)] = torch.tensor(0)
#             break
#         else:
#             mask[action] = mask[action].clone() * 0.3
#
#             event = new_mapping(action)
#             events.append(event)
#             mini_actions, _, comp_vals, conds, actions_vals, _, comps = model.get_cols_mini_actions(data.cuda())
#             all_comps.append(comps)
#             for j, action_val in enumerate(actions_vals):
#                 data = data.clone()
#                 try:
#                     data[data_size + count * (model.num_cols + 1) + j + 1] = model.embedding_actions(
#                         torch.tensor(action_val))
#                     data2[data_size + count * (model.num_cols + 1) + j + 1] = data[data_size + count * (model.num_cols + 1) + j + 1]
#                 except Exception as e:
#                     print(f"count {count}, j {j}")
#             actions.append(mini_actions)
#
#             currentPath = pathlib.Path(os.path.dirname(__file__))
#             absolutePath = str(currentPath.parent)
#             sys.path.append(absolutePath)
#             file = os.path.join(absolutePath, "Model", "training", "{}.txt".format(i))
#
#             # comp_vals = set_values(comp_vals, model.all_cols, mini_actions, event, conds, file)
#             all_conds.append(conds)
#
#
#             if comp_vals.count("nop") != len(comp_vals):
#                 # bayesian_dict = set_values_bayesian(comp_vals, model.all_cols, mini_actions, event, all_conds, file)
#                 bayesian_dict = set_values_bayesian(comp_vals,
#                     model.all_cols, mini_actions, event,
#                     all_conds, file, model.max_values_bayes,
#                      model.min_values_bayes
#                  )
#                 store_to_file(events, actions, i, comp_values, model.cols, all_conds, comp_vals, all_comps)
#                 b_optimizer = BayesianOptimization(
#                     f=bayesian_function,
#                     pbounds=bayesian_dict,
#                     random_state=1,
#                 )
#                 try:
#                     b_optimizer.maximize(
#                         init_points=5,
#                         n_iter=3,
#                     )
#
#                     selected_values = list(b_optimizer.max['params'].values())
#                 except Exception as e:
#                     print(bayesian_dict)
#                     selected_values = [max(model.normailze_values) for _ in range(len(bayesian_dict))]
#                 comp_vals = replace_values(comp_vals, selected_values)
#
#             comp_values.append(comp_vals)
#             # all_conds.append(conds)
#             pattern = OpenCEP_pattern(
#                 events, actions, i, comp_values, model.cols, all_conds, all_comps
#             )
#             eff_pattern = pattern.condition
#             with open("Data/Matches/{}Matches.txt".format(i), "r") as f:
#                 pattern_copy = (data[-added_info_size:]).detach().cpu().numpy().reshape(1,-1)
#                 rating = model.knn.predict(pattern_copy).item()
#
#                 reward = int(f.read().count("\n") / (len(actions) + 1))
#                 if 1:
#                     reward *= rating
#                     ratings.append(rating)
#                     rewards.append(reward)
#                     sys.stdout.write(f"Knn out: {rating}\n")
#                 str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols, all_comps)
#             if count >= model.match_max_size:
#                 is_done = True
#
#     if len(ratings) == 0:
#         return [], [], -1, " "
#     else:
#         events_ball = [4 if event in [4,8,10] else event for event in events]
#         if len(np.unique(events_ball)) == 1:
#             ratings[-1] = 0
#         return events, data[-added_info_size:], ratings[-1], str_pattern
#
#
# def predict_patterns(model):
#     def choose_best_pattern(patterns, ratings):
#         def similar(a, b):
#             return SequenceMatcher(None, a, b).ratio()
#
#         results = [0.0] * len(ratings)
#         for i in range(0, len(ratings) - 1):
#             for j in range(i, len(events)):
#                 sim = similar(patterns[i], patterns[j])
#                 results[i] += sim
#                 results[j] += sim
#         results = torch.tensor(results) * torch.tensor(ratings)
#         print(f"The similarities are:{results}")
#
#         best_index = np.argmax(results)
#         return best_index, results[best_index]
#
#     model.eval()
#     events = []
#     patterns = []
#     ratings = []
#     pattern_strs = []
#     # types = [] Todo: do this also
#     for i, data in enumerate(model.data[-100:]):
#         event, pattern, rating, pattern_str = predict_window(model, i, data)
#         if len(event) != 0:
#             # event = [str(val) for val in event]
#             events.append(event)
#             pattern = ",".join([str(i) for i in pattern])
#             patterns.append(pattern)
#             ratings.append(rating)
#             pattern_strs.append(pattern_str)
#
#
#     print("Looking for most similar\n")
#     best_index, best_sim = choose_best_pattern(patterns, ratings)
#     print(best_sim)
#     print(events[best_index])
#     print(pattern_strs[best_index])


def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(len(costs), dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def bayesian_hidden_size_search(hidden_size):
    hidden_size = int(hidden_size)
    global class_inst
    train_inst = deepcopy(class_inst)
    train_inst.layers_based_hidden(hidden_size)
    return train(train_inst, num_epochs=5,  bs=200 , split_factor=49/100)

def bayesian_lr_search(lr):
    global class_inst
    train_inst = deepcopy(class_inst)
    train_inst.lr = lr
    train_inst.optimizer = torch.optim.Adam(train_inst.parameters(), lr=lr)
    print(train_inst.parameters())
    # train_inst.layers_based_hidden(hidden_size)
    return train(train_inst, num_epochs=5,  bs=200 , split_factor=49/100)


def bayesian_hidden2_size_search(hidden_size):
    hidden_size = int(hidden_size)
    global class_inst
    train_inst = deepcopy(class_inst)
    train_inst.hidden_size2 = hidden_size
    train_inst.layers_based_hidden2(hidden_size)
    return train(train_inst, num_epochs=5,  bs=200 , split_factor=49/100)


def bayesian_hidden_search(hidden_size1, hidden_size2):
    global class_inst
    train_inst = deepcopy(class_inst)
    hidden_size1 = int(hidden_size1)
    hidden_size2 = int(hidden_size2)
    train_inst.layers_hidden(hidden_size1, hidden_size2)
    return train(train_inst, num_epochs=5,  bs=200 , split_factor=49/100)


def main(parser):
    args = parser.parse_args()
    max_vals = [int(i) for i in args.max_vals.split(",")]
    norm_vals = [int(i) for i in args.norm_vals.split(",")]
    all_cols = args.all_cols.replace(" ", "").split(",")
    eff_cols = args.eff_cols.replace(" ", "").split(",")
    if args.pattern_path == "":
        rating_flag = False
    else:
        rating_flag = True

    results = {}
    data = None
    first = True
    suggested_models = []
    all_patterns = []

    # for split_factor in range(60, 90, 10):
    # if 1:
    for window_size in range(550, 650, 50):
    # if 1:
        split_factor = 54
        eff_split_factor = split_factor / 100
        global class_inst
        class_inst = ruleMiningClass(data_path=args.data_path,
                                    pattern_path=args.pattern_path,
                                    events_path=args.events_path,
                                    num_events=args.num_events,
                                    match_max_size=args.max_size,
                                    window_size=window_size,
                                    max_fine_app=args.max_fine_app,
                                    max_values=max_vals,
                                    normailze_values=norm_vals,
                                    all_cols=all_cols,
                                    eff_cols=eff_cols,
                                    lr=args.lr,
                                    max_time=args.pattern_max_time,
                                    hidden_size1=512,
                                    hidden_size2=1000,
                                    init_flag=True)
        #
        # optimizer = BayesianOptimization(
        #     f=bayesian_hidden_search,
        #     pbounds={'hidden_size1': (512, 2048), 'hidden_size2': (512, 2048)},
        #     random_state=1,
        # )
        # optimizer.maximize(
        #     init_points=3,
        #     n_iter=5,
        # )

        result = train(class_inst, num_epochs=args.epochs, bs=args.bs, split_factor=eff_split_factor, rating_flag=rating_flag)
        # result, patterns = train(class_inst, num_epochs=args.epochs, bs=args.bs, split_factor=eff_split_factor, rating_flag=rating_flag)
    #     all_patterns.append(patterns)
    #     cuda_handle.empty_cache()
    #     print(patterns)
    #     results.update({split_factor: result})
    #     suggested_models.append({split_factor: class_inst})
    # print(results)
    # pareto_results = np.array(list(results.values()))
    # pareto_results = np.array([np.array(list(res.values())) for res in pareto_results])
    # print(pareto_results)
    # patero_results = is_pareto_efficient(pareto_results)
    # good_patterns = []
    # for patero_res, model, patterns in zip(patero_results, suggested_models, all_patterns):
    #     if patero_res:
    #         print(model)
    #         good_patterns.extend(list(patterns.values()))
    #         print(patterns)
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CEP pattern miner')
    parser.add_argument('--bs', default=200, type=int, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, help='num epochs to train')
    parser.add_argument('--lr', default=1e-5, type=float, help='starting learning rate')
    parser.add_argument('--hidden_size', default=4096, type=int, help='hidden_size param for model')
    parser.add_argument('--max_size', default=8, type=int, help='max size of pattern')
    parser.add_argument('--max_fine_app', default=55, type=int, help='max appearance of pattnern in a single window')
    parser.add_argument('--pattern_max_time', default=7, type=int, help='maximum time for pattern (seconds)')
    parser.add_argument('--window_size', default=350, type=int, help='max size of input window')
    parser.add_argument('--num_events', default=41,type=int, help='number of unique events in data')
    parser.add_argument('--data_path', default='Football/xaa', help='path to data log')
    parser.add_argument('--events_path', default='Football/events', help='path to list of events')
    parser.add_argument('--pattern_path', default='Patterns/pattern16.csv', help='path to known patterns')
    parser.add_argument('--final_data_path', default='Football/xal', help='path to next level data')
    parser.add_argument('--max_vals', default = "97000, 100000, 25000, 20000, 20000, 20000", type=str, help="max values in columns")
    parser.add_argument('--norm_vals', default = "24000, 45000, 6000, 9999, 9999, 9999", type=str, help="normalization values in columns")
    parser.add_argument('--all_cols', default = 'x, y, z, vx, vy, vz, ax, ay, az', type=str, help="all cols in data")
    parser.add_argument('--eff_cols', default = 'x, y, z, vx, vy', type=str, help="cols to use in model")
    torch.set_num_threads(50)
    main(parser)
