import argparse
import torch
import random
import torch.nn as nn
import os
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

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
    check_predictor,
    calc_near_windows,
)

from Model.rating_module import (
    rating_main,
    ratingPredictor,

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
import json
from torch.optim.lr_scheduler import StepLR

from models import ActorCriticModel

GRAPH_VALUE = 50
GAMMA = 0.99
EMBDEDDING_TOTAL_SIZE = 8
PAD_VALUE = -5.5
class_inst = None
num_epochs_trained = None
total_steps_trained = 0

with torch.autograd.detect_anomaly():
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
            lr_actor=1e-6,
            lr_critic=1e-6,
            init_flag=False,
            hidden_size1=512,
            hidden_size2=2048,
            exp_name="Football"
        ):
            super().__init__()
            # self.lr = lr
            self.exp_name = exp_name
            self.actions = [">", "<", "="]
            self.max_predict = (match_max_size + 1) * (len(eff_cols) + 1)
            self.events = np.loadtxt(events_path, dtype='str')
            self.num_events = len(self.events)
            self.match_max_size = match_max_size
            self.max_values = max_values
            self.window_size = window_size
            self.normailze_values = normailze_values
            self.embedding_events = nn.Embedding(num_events + 1, 3)
            self.embedding_values = [nn.Embedding(max_val, 3) for max_val in max_values]
            self.pattern_path = pattern_path.split("/")[-1].split(".")[0]
            if init_flag:
                if not os.path.exists(f"Processed_Data/{self.exp_name}/{self.window_size}.pt"):
                    self.data = self._create_data(data_path)
                    self.data = self.data.view(len(self.data), -1)
                    self.data = self.data.detach().clone().requires_grad_(True)
                    torch.save(self.data, f"Processed_Data/{self.exp_name}/{self.window_size}.pt")
                else:
                    self.data = torch.load(f"Processed_Data/{exp_name}/{self.window_size}.pt").requires_grad_(True)
            global EMBDEDDING_TOTAL_SIZE
            if self.exp_name == "StarPilot":
                EMBDEDDING_TOTAL_SIZE = 8
            elif self.exp_name == "GPU":
                EMBDEDDING_TOTAL_SIZE = 99
            else:
                raise Exception("Data set not supported!")
            self.hidden_size1 = hidden_size1
            self.hidden_size2 = hidden_size2
            self.num_cols = len(eff_cols)
            self.num_actions = (len(self.actions) * (self.match_max_size + 1)) * 2 * 2 + 1  # [>|<|= * [match_max_size + 1(value)] * not / reg] * (and/or) |nop
            self.embedding_actions = nn.Embedding(self.num_actions + 1, 1)
            self.embedding_desicions = nn.Embedding(self.num_events + 1, 1)
            self.actor_critic = ActorCriticModel(
                                num_events=self.num_events,
                                match_max_size=self.match_max_size,
                                window_size=self.window_size,
                                num_cols=self.num_cols,
                                hidden_size1=self.hidden_size1,
                                hidden_size2=self.hidden_size2,
                                embeddding_total_size=EMBDEDDING_TOTAL_SIZE
                                )

            self._create_training_dir(data_path)
            print("finished training dir creation!")

            params = list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters())

            self.critic_optimizer = torch.optim.SGD(params, lr=lr_critic)

            self.actor_optimizer = torch.optim.SGD(self.actor_critic.actor.parameters(), lr=lr_actor)
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
            df.rating = df.rating.apply(lambda x : min(round(float(x) - 1), 49))
            # print(os.path.exists(f"Processed_knn/{self.pattern_path}"))
            if not os.path.exists(f"Processed_knn/{self.pattern_path}"):
                print("Creating Knn!")
                os.mkdir(f"Processed_knn/{self.pattern_path}")
                # str_list_columns = ["conds", "actions"]
                str_list_columns = ["actions"]
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

                    if not os.path.exists(f"Processed_knn/{self.pattern_path}/dicts/"):
                        os.mkdir(f"Processed_knn/{self.pattern_path}/dicts/")
                    with open(f"Processed_knn/{self.pattern_path}/dicts/{len(self.list_of_dfs)}", 'w') as fp:
                        json.dump(add_df, fp)

                    combined = fix_str_list_columns(temp)
                    combined.columns = list(map(lambda x: col + "_" + str(x), combined.columns))

                    if df_new is None:
                        df_new = combined
                    else:
                        df_new = pd.concat([df_new, combined], axis=1).reset_index(drop=True)
                    df_new = df_new.fillna(PAD_VALUE)


                df_new.to_csv(f"Processed_knn/{self.pattern_path}/df", index=False)

            else:
                file_names = os.listdir(f"Processed_knn/{self.pattern_path}/dicts")
                for file_name in file_names:
                    with open(f"Processed_knn/{self.pattern_path}/dicts/{file_name}", "r") as read_file:
                        self.list_of_dfs.append(json.load(read_file))
                df_new = pd.read_csv(f"Processed_knn/{self.pattern_path}/df")


            knn = KNN(n_neighbors=5)
            knn.fit(df_new, df["rating"])
            self.knn_avg = df.rating.mean()

            test_pred = ratingPredictor(df_new, df["rating"])
            self.pred_optim = torch.optim.Adam(params=test_pred.parameters(), lr=1e-4)
            self.pred_sched = StepLR(self.pred_optim, step_size=3000, gamma=0.3)

            if not os.path.exists(f"Processed_knn/{self.pattern_path}/rating_model.pt"):
                test_pred._train(self.pred_optim, self.pred_sched, count=0, max_count=10, max_total_count=100, n=10)
                torch.save(test_pred, f"Processed_knn/{self.pattern_path}/rating_model.pt")
            else:
                test_pred = torch.load(f"Processed_knn/{self.pattern_path}/rating_model.pt")
                #TODO: remove!
                # test_pred._train(self.pred_optim, self.pred_sched, count=0, max_count=5, max_total_count=50, n=0)

            print(len(test_pred.ratings_col_train))
            self.pred_pattern = test_pred
            self.pred_pattern.rating_df_unlabeld = None
            self.pred_pattern.unlabeld_strs = []
            return knn

        def _create_data(self, data_path):
            date_time_obj = None
            all_data = None
            if self.exp_name in ["Football", "GPU"]:
                data = None
                first_line = True
                with open(data_path) as f:
                    for count, line in enumerate(f):
                        if first_line and self.exp_name == "GPU":
                            first_line = False
                            continue
                        if count % 10000 == 0:
                            print(count)

                        values = line.split("\n")[0]
                        values = values.split(",")
                        if self.exp_name == "Football":
                            index = 0
                        else:
                            index = -1
                        event = values[index]
                        event = self.embedding_events(torch.tensor(int(new_mapping(event, self.events, reverse=True))))
                        if self.exp_name == "Football":
                            values = values[2:] # skip sid and ts
                        else:
                            values = values[1:] # skip sid and ts
                            values = values[:-1]
                        # print(values)
                        # print(len(values))
                        try:
                            embed_values = [self.embedding_values[i](torch.tensor(int(float(value) * 100) + self.normailze_values[i])) for (i,value) in enumerate(values[:len(self.normailze_values)])]
                            embed_values.insert(0, event)
                        except Exception as e:
                            embed_values = []
                            for i, value in enumerate(values[:len(self.normailze_values)]):
                                value = float(value)
                                value *= 100
                                a = self.normailze_values[i]
                                a = torch.tensor(int(value) + a)
                                a = self.embedding_values[i](a)
                                embed_values.append(a)
                        if data is None:
                            data = torch.cat(tuple(embed_values), 0)
                            data = data.unsqueeze(0)
                        else:
                            new_data = torch.cat(tuple(embed_values), 0)
                            new_data = new_data.unsqueeze(0)
                            data = torch.cat((data, new_data), 0)

                sliding_window_data = None
                count = 0
                for i in range(0, len(data) - self.window_size, self.window_size // 10):
                    count += 1
                    if count % 100 == 0:
                        print(count)
                    if sliding_window_data is None:
                        sliding_window_data = data[i : i + self.window_size]
                        sliding_window_data = sliding_window_data.unsqueeze(0)
                    else:
                        to_add = data[i : i + self.window_size].unsqueeze(0)
                        sliding_window_data = torch.cat((sliding_window_data, to_add))

                all_data = sliding_window_data

            elif self.exp_name == "StarPilot":
                files = os.listdir(data_path)[:500]
                for file in files:
                    data = None
                    sliding_window_data = None

                    with open(os.path.join(data_path,file)) as f:
                        for line in f:
                            values = line.split("\n")[0]
                            values = values.split(",")
                            event = values[1]
                            event = self.embedding_events(torch.tensor(int(new_mapping(event, self.events, reverse=True))))
                            event = event.detach().numpy()
                            values = values[2:] # skip sid and ts
                            values = np.concatenate((event, values))
                            values = [float(val) for val in values]
                            if data is None:
                                data = torch.tensor(values)
                                data = data.unsqueeze(0)
                            else:
                                new_data = torch.tensor(values)
                                new_data = new_data.unsqueeze(0)
                                data = torch.cat((data, new_data), 0)


                    for i in range(0, len(data) - self.window_size):
                        if sliding_window_data is None:
                            sliding_window_data = data[i : i + self.window_size]
                            sliding_window_data = sliding_window_data.unsqueeze(0)
                        else:
                            to_add = data[i : i + self.window_size].unsqueeze(0)
                            sliding_window_data = torch.cat((sliding_window_data, to_add))

                    if all_data is None:
                        all_data = sliding_window_data
                    else:
                        all_data = torch.cat((all_data, sliding_window_data))
            else:
                raise Exception("Data set not supported!")

            return all_data

        def _create_training_dir(self, data_path):
            if not os.path.exists("Model/training/"):
                os.mkdir("Model/training/")
            lines = []
            if self.exp_name in ["Football", "GPU"]:
                with open(data_path) as f:
                    for line in f:
                        lines.append(line)

                count = 0
                for i in range(0, len(lines) - self.window_size, self.window_size // 10):
                    with open("Model/training/{}.txt".format(count), "w") as f:
                        for j in range(i, i + self.window_size):
                            f.write(lines[j])
                    count += 1

            elif self.exp_name == "StarPilot":
                current_files_created = 0
                files = os.listdir(data_path)[:200]
                for file in files:
                    lines = []
                    with open(os.path.join(data_path, file)) as f:
                        for line in f:
                            lines.append(line)

                        for i in range(0, len(lines) - self.window_size):
                            with open(f"Model/training/{str(current_files_created)}.txt", "w") as new_file:
                                for j in range(i, i + self.window_size):
                                    new_file.write(lines[j])
                            current_files_created += 1
            else:
                raise Exception("Data set not supported")

        def forward(self, input, old_desicions, training_factor=0.0):
            base_value, event_after_softmax = self.actor_critic.forward_actor(input, old_desicions, training_factor=training_factor)
            value_reward, value_rating = self.actor_critic.forward_critic(base_value)
            return event_after_softmax, value_reward, value_rating

        def get_event(self, input, old_desicions, index=0, training_factor=0.0):
            global total_steps_trained
            total_steps_trained += 1
            probs, value_reward, value_rating = self.forward(Variable(input), Variable(old_desicions), training_factor=training_factor)
            numpy_probs = probs.detach().cpu().numpy()
            action = None
            numpy_probs = np.squeeze(numpy_probs).astype(float)
            numpy_probs = numpy_probs / np.sum(numpy_probs)
            
            try:
                action = np.random.multinomial(
                    n=1, pvals=numpy_probs, size=1
                )
                num_actions = len(numpy_probs)
                action = np.argmax(action)
            except Exception as e:
                print(e)
                print("----")
                print(numpy_probs)

                print("----")
                print(probs)
                exit(0)
            entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2
            if np.random.rand() > 1 - training_factor:
                action = np.random.randint(num_actions)
            if index % 50 == 0:
                print(probs)

            log_prob = torch.log(probs.squeeze(0)[action])
            if abs(log_prob) < 0.1:
                self.count += 1

            return action, log_prob, value_reward, value_rating, entropy

        def single_col_mini_action(self, data, index, training_factor=0.0):
            highest_prob_value = None
            highest_prob_action, log_prob, entropy = self.actor_critic.forward_actor_mini_actions(
                index, data, training_factor
            )
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

            base_value, _ = self.actor_critic.forward_actor(data, old_desicions.cuda(), False, training_factor)
            value_reward, value_rating = self.actor_critic.forward_critic(base_value)
            for i in range(self.num_cols):
                action, value, log, entropy = self.single_col_mini_action(base_value, i, training_factor) #this is weird, should update data after actions
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
            return mini_actions, log_probs, compl_vals, conds, mini_actions_vals, total_entropy, comps_to, value_reward, value_rating



    def update_policy(policy_network, ratings, rewards, log_probs, values_rating, values_reward,
                    Qval_rating, Qval_reward, entropy_term, epoch_idx, flag=False):

        def l1_penalty(log_probs, l1_lambda=0.001):
            """
            Returns the L1 penalty of the params.
            """
            l1_norm = sum(log_prob.abs().sum() for log_prob in log_probs)
            return l1_lambda*l1_norm / len(log_probs)

        Qvals_reward = np.zeros_like(values_reward)
        Qvals_rating = np.zeros_like(values_rating)
        for t in reversed(range(len(rewards))):
            Qval_0 = rewards[t] + GAMMA * Qval_reward
            Qval_1 = ratings[t] + GAMMA * Qval_rating
            Qvals_reward[t] = Qval_0
            Qvals_rating[t] = Qval_1

        values = torch.FloatTensor((values_reward, values_rating)).requires_grad_(True)
        Qvals = torch.FloatTensor((Qvals_reward, Qvals_rating)).requires_grad_(True)
        log_probs = torch.stack(log_probs).requires_grad_(True)
        advantage = Qvals - values
        advantage = advantage.to(log_probs.device)
        log_probs_reg = l1_penalty(log_probs, l1_lambda=0.05)

        actor_loss = (-log_probs * advantage).mean().requires_grad_(True)

        actor_loss += log_probs_reg
        critic_loss = 0.5 * advantage.pow(2).mean().requires_grad_(True)
        policy_network.actor_optimizer.zero_grad()
        policy_network.critic_optimizer.zero_grad()
        actor_loss_1 = actor_loss.cpu().detach().numpy()
        actor_loss = actor_loss.cuda()
        critic_loss_1 = critic_loss.cpu().detach().numpy()
        critic_loss = critic_loss.cuda()
        if flag:
            actor_loss.backward()
            policy_network.actor_optimizer.step()
        else:
            critic_loss.backward()
            policy_network.critic_optimizer.step()

        return actor_loss_1, critic_loss_1

    def train(model, num_epochs=5, test_epcohs=False, split_factor=0, bs=0, rating_flag=True):
        run_name = f"GPU data first test"
        not_finished_count = 0
        run = wandb.init(project='Pattern_Mining', entity='guyshapira', name=run_name, settings=wandb.Settings(start_method='fork'))
        config = wandb.config
        config.hidden_size1 = model.hidden_size1
        config.hidden_size2 = model.hidden_size2
        config.current_epoch = 0
        # config.learning_rate = model.lr
        config.batch_size = bs
        config.window_size = model.window_size
        config.num_epochs = num_epochs
        config.split_factor = split_factor
        config.total_number_of_steps = total_steps_trained

        flatten = lambda list_list: [item for sublist in list_list for item in sublist]
        torch.autograd.set_detect_anomaly(True)
        added_info_size = (model.match_max_size + 1) * (model.num_cols + 1)
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
        total_count = -5
        count_actor = 0
        count_critic = 0

        global num_epochs_trained
        for epoch in range(num_epochs):
            config.update({"current_epoch" : epoch}, allow_val_change=True)

            model.count = 0
            if num_epochs_trained is None:
                num_epochs_trained = 0
            else:
                num_epochs_trained += 1
            print(f"Not finished = {not_finished_count}\n")
            currentPath = pathlib.Path(os.path.dirname(__file__))
            absolutePath = str(currentPath.parent)
            sys.path.append(absolutePath)

            with tqdm.tqdm(total=bs, file=pbar_file) as pbar:
                in_round_count = 0
                path = os.path.join(absolutePath, "Model", "training")
                data_len = len(os.listdir(path))
                for index in range(epoch + 2, min(data_len - 2, len(model.data)), data_len // bs):
                    set_data = None
                    if total_count >= bs:
                        total_count = 0
                        turn_flag = 1 - turn_flag

                    if total_count >= switch_flag:
                        turn_flag = 1 - turn_flag

                    if in_round_count >= bs:
                        break
                    total_count += 1
                    in_round_count += 1
                    data = model.data[index]
                    data_size = len(data)
                    old_desicions = torch.tensor([PAD_VALUE] * added_info_size)
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
                    values_rating, values_reward, comp_values, patterns, ratings = (
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                    normalize_rating, normalize_reward = [], []
                    if total_count % 250 == 0 and training_factor > 0.3:
                        training_factor /= 1.2
                    while not is_done:
                        if not set_data is None:
                            data = set_data.clone().detach().requires_grad_(True)
                            set_data = None
                        data = data.cuda()
                        action, log_prob, value_reward, value_rating, entropy = model.get_event(
                            data, old_desicions, in_round_count, training_factor=training_factor
                        )
                        old_desicions = old_desicions.clone()
                        old_desicions[count * (model.num_cols + 1)] = model.embedding_desicions(
                            torch.tensor(action)
                        ).cuda()
                        count += 1
                        value_rating = value_rating.detach().cpu().numpy()[0]
                        value_reward = value_reward.detach().cpu().numpy()[0]
                        values_rating.append(value_rating)
                        values_reward.append(value_reward)
                        entropy_term += entropy

                        if action == model.num_events:
                            ratings.append(1)
                            log_probs.append(log_prob)

                            if len(actions) == 0:
                                rewards.append(-1.5)
                                real_rewards.append(-1.5)
                            else:
                                rewards.append(10)
                                real_rewards.append(10)
                                break
                        else:
                            event = new_mapping(action, model.events)
                            events.append(event)
                            mini_actions, log, comp_vals, conds, actions_vals, entropy, comps_to, value_reward, value_rating = \
                                model.get_cols_mini_actions(data, old_desicions, training_factor=training_factor)
                            all_comps.append(comps_to)
                            entropy_term += entropy
                            for j, action_val in enumerate(actions_vals):
                                old_desicions = old_desicions.clone()
                                old_desicions[count * (model.num_cols + 1) + j + 1] = model.embedding_actions(torch.tensor(action_val))
                            
                            log_prob = (log_prob + log.item()) / 2
                            log_probs.append(log_prob)
                            actions.append(mini_actions)


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
                                        init_points=10,
                                        n_iter=0,
                                    )
                                    selected_values = [round(selected_val, 3) for selected_val in b_optimizer.max['params'].values()]
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
                            rating, norm_rating = rating_main(model, events, all_conds, actions, str_pattern, rating_flag, epoch, pred_flag=True)

                            #TODO: remove, added for sacle factor testing
                            # rating /= 5

                            ratings.append(rating)
                            normalize_rating.append(norm_rating)

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
                                    if reward >= model.max_fine_app:
                                        reward = 2 * model.max_fine_app - reward
                                    global EMBDEDDING_TOTAL_SIZE
                                    if reward != 0:
                                        try:
                                            first_ts = content.split("\n")[0].split("ts\': ")[1].split(",")[0]
                                            with open(file, "r") as data_file:
                                                content = data_file.read()
                                            lines = content.split("\n")
                                            row_idx = 0
                                            for num, line in enumerate(lines):
                                                if line.startswith(first_ts):
                                                    row_idx = num
                                                    break
                                            chunk_size = EMBDEDDING_TOTAL_SIZE * row_idx
                                            set_data = torch.tensor([i for i in data])
                                            set_data[: chunk_size] = torch.tensor([PAD_VALUE] * chunk_size)
                                        except Exception as e:
                                            print(e)
                                            print(f"Reward : {reward}")
                                            raise(e)

                                    try:
                                        near_windows_rewards = calc_near_windows(index, pattern, len(actions),
                                            model.max_fine_app, model.window_size, data_len)
                                        reward = reward * 0.75 + near_windows_rewards * 0.25

                                    except Exception as e:
                                        print(e)
                                        pass
                                    real_rewards.append(reward)
                                    # if reward == 0 and turn_flag:
                                    #     normalize_reward.append(-25)
                                    #     rewards.append(-1.5)
                                    #     break
                                    normalize_reward.append(reward - 20)
                                    #TODO: Remove this!
                                    # reward *= rating
                                    reward *= (rating / 5)
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

                    _, Qval_reward, Qval_rating = model.forward(data, torch.tensor([PAD_VALUE] * added_info_size), training_factor=training_factor)
                    Qval_rating = Qval_rating.detach().cpu().numpy()[0]
                    Qval_reward = Qval_reward.detach().cpu().numpy()[0]

                    del data
                    gc.collect()
                    if turn_flag == 0:
                        send_rewards = ratings
                    else:
                        send_rewards = real_rewards

                    actor_flag = False
                    if count_actor < 100:
                        actor_flag = True
                        count_actor += 1
                    elif count_critic < 250:
                        count_critic += 1
                    else:
                        count_actor = 0
                        count_critic = 0
                    a1, c1  = update_policy(model, ratings, real_rewards, log_probs, values_rating, values_reward,
                                                                    Qval_rating, Qval_reward,
                                                                    entropy_term, epoch, flag=actor_flag)


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

                    if True:
                        sys.stdout.write(
                            "\nReal reward : {}, Rating {}, Max Rating : {},  comparisons : {}\n".format(
                                real_rewards[index_max],
                                ratings[index_max],
                                np.max(ratings),
                                sum([t != "nop" for sub in comp_values for t in sub]),
                            )
                        )
                        if (real_rewards[index_max] > 2 or random.randint(0,3) > 1) or (ratings[index_max] > 2 or random.randint(0,3) > 1):
                            wandb.log({"reward": real_rewards[index_max], "rating": ratings[index_max],
                                    "max rating": np.max(ratings), "actor_flag": int(actor_flag),
                                    "actor_loss_reward": a1, "critic_loss_reward": c1,
                                    "curent_step": total_steps_trained})

                        # if total_steps_trained > 4500:
                        #     # Only for sweeps!
                        #     return None, None


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
                    config.update({"total_number_of_steps" : total_steps_trained}, allow_val_change=True)
                    if model.count > 100:
                        print("\n\n\n---- Stopping early because of low log ----\n\n\n")
                        model.count = 0
                        for g1, g2 in zip(model.actor_optimizer.param_groups, model.critic_optimizer.param_groups):
                            g1['lr'] *= 0.95
                            g2['lr'] *= 0.95
                        # continue
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

                # for sweeps on newton
                if 0:
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
                if epoch < 2:
                    model.pred_pattern.save_all()
                    # model.pred_pattern.train()
                    # model.pred_optim = torch.optim.Adam(params=model.pred_pattern.parameters(), lr=5e-5)
                    # model.pred_pattern._train(model.pred_optim, None, count=0, max_count=5, max_total_count=50, n=3, retrain=True)
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
        # return best_res
        return best_res, best_found



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


    def main(parser):
        args = parser.parse_args()
        max_vals = [int(float(i) * 100 * 1.1) for i in args.max_vals.split(",")]
        norm_vals = [int(float(i) * 100 * 1.1) for i in args.norm_vals.split(",")]
        all_cols = [item for item in args.all_cols.split(',')]

        eff_cols = [item for item in args.eff_cols.split(',')]
        if args.pattern_path == "":
            rating_flag = False
        else:
            rating_flag = True

        results = {}
        data = None
        first = True
        suggested_models = []
        all_patterns = []
        split_factor = args.split_factor
        for window_size in [args.window_size]:
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
                                        max_time=args.pattern_max_time,
                                        lr_actor=args.lr_actor,
                                        lr_critic=args.lr_critic,
                                        hidden_size1=args.hidden_size1,
                                        hidden_size2=args.hidden_size2,
                                        exp_name=args.exp_name,
                                        init_flag=True)

            if 1:
                result, patterns = train(class_inst, num_epochs=args.epochs, bs=args.bs, split_factor=split_factor, rating_flag=rating_flag)
                all_patterns.append(patterns)
                cuda_handle.empty_cache()
                print(patterns)
                results.update({split_factor: result})
                suggested_models.append({split_factor: class_inst})

        if 0:
            print(results)
            pareto_results = np.array(list(results.values()))
            pareto_results = np.array([np.array(list(res.values())) for res in pareto_results])
            print(pareto_results)
            patero_results = is_pareto_efficient(pareto_results)
            good_patterns = []
            for patero_res, model, patterns in zip(patero_results, suggested_models, all_patterns):
                if patero_res:
                    print(model)
                    good_patterns.extend(list(patterns))
                    print(patterns)


            run_OpenCEP(events=args.final_data_path, patterns=good_patterns, test_name="secondLevel27April")

        return



    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='CEP pattern miner')
        parser.add_argument('--bs', default=500, type=int, help='batch size')
        parser.add_argument('--epochs', default=20, type=int, help='num epochs to train')
        parser.add_argument('--lr_actor', default=5e-8, type=float, help='starting learning rate for actor')
        parser.add_argument('--lr_critic', default=5e-6, type=float, help='starting learning rate for critic')
        parser.add_argument('--hidden_size1', default=1024, type=int, help='hidden_size param for model')
        parser.add_argument('--hidden_size2', default=2048, type=int, help='hidden_size param for model')
        parser.add_argument('--max_size', default=8, type=int, help='max size of pattern')
        parser.add_argument('--max_fine_app', default=150, type=int, help='max appearance of pattnern in a single window')
        parser.add_argument('--pattern_max_time', default=100, type=int, help='maximum time for pattern (seconds)')
        parser.add_argument('--window_size', default=700, type=int, help='max size of input window')
        parser.add_argument('--num_events', default=41, type=int, help='number of unique events in data')
        parser.add_argument('--split_factor', default=0.2, type=float, help='split how much train to rating and how much for reward')
        parser.add_argument('--data_path', default='GPU/gpu_test.csv', help='path to data log')

        parser.add_argument('--events_path', default='GPU/Events', help='path to list of events')


        parser.add_argument('--pattern_path', default='', help='path to known patterns')
        parser.add_argument('--final_data_path', default='', help='path to next level data')
        parser.add_argument('--max_vals', default = "10570.0, 390.99, 246.81, 250.25, 10544.0, 415.56, 245.21, 251.9, 10964.0, 382.53, 220.62, 247.52, 10982.0, 390.3, 218.825, 245.96, 10980.0, 386.36, 209.95, 246.64, 11004.0, 375.14, 231.67, 248.29, 11002.0, 410.72, 222.85, 248.08, 10982.0, 414.95, 216.07, 245.76", type=str, help="max values in columns")
        parser.add_argument('--norm_vals', default = "1.0, 19.02, 18.17, 0.0, 1.0, 11.13, 10.29, 0.0, 1.0, 10.25, 9.93, 0.0, 1.0, 7.36, 7.12, 0.0, 1.0, 8.59, 8.04, 0.0, 1.0, 8.36, 7.7, 0.0, 1.0, 7.01, 6.91, 0.0, 1.0, 1.16, 0.91, 0.0", type=str, help="normalization values in columns")

        parser.add_argument('--all_cols','--list',  default = "FB Memory Usage Used GPU_0,Power Samples Max GPU_0,"\
                            "Power Samples Min GPU_0,Power Samples Avg GPU_0,"\
                            "FB Memory Usage Used GPU_1,Power Samples Max GPU_1,"\
                            "Power Samples Min GPU_1,Power Samples Avg GPU_1,"\
                            "FB Memory Usage Used GPU_2,Power Samples Max GPU_2,"\
                            "Power Samples Min GPU_2,Power Samples Avg GPU_2,"\
                            "FB Memory Usage Used GPU_3,Power Samples Max GPU_3,"\
                            "Power Samples Min GPU_3,Power Samples Avg GPU_3,"\
                            "FB Memory Usage Used GPU_4,Power Samples Max GPU_4,"\
                            "Power Samples Min GPU_4,Power Samples Avg GPU_4,"\
                            "FB Memory Usage Used GPU_5,Power Samples Max GPU_5,"\
                            "Power Samples Min GPU_5,Power Samples Avg GPU_5,"\
                            "FB Memory Usage Used GPU_6,Power Samples Max GPU_6,"\
                            "Power Samples Min GPU_6,Power Samples Avg GPU_6,"\
                            "FB Memory Usage Used GPU_7,Power Samples Max GPU_7,"\
                            "Power Samples Min GPU_7,Power Samples Avg GPU_7", 
                            type=str, help="all cols in data")

        parser.add_argument('--eff_cols','--list1', default = "FB Memory Usage Used GPU_0,Power Samples Max GPU_0,"\
                "Power Samples Min GPU_0,Power Samples Avg GPU_0", 
                type=str, help="all cols in data")
        parser.add_argument('--exp_name', default = 'GPU', type=str)
        torch.set_num_threads(50)
        main(parser)
