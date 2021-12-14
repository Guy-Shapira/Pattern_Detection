import argparse
from math import log, sqrt
import torch
import random
import torch.nn as nn
import os
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
from distutils.util import strtobool

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
    mapping_for_baseline,
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
import json
from torch.optim.lr_scheduler import StepLR

from models_baseline import ActorCriticModel

GRAPH_VALUE = 50
GAMMA = 0.99
EMBDEDDING_TOTAL_SIZE = 8
PAD_VALUE = -5.5
class_inst = None
num_epochs_trained = None
total_steps_trained = 0

# torch.manual_seed(50)
# random.seed(50)
# np.random.seed(50)
torch.cuda.set_device(7)

with torch.autograd.set_detect_anomaly(True):
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
            exp_name="Football",
            knowledge_flag=True,
            run_mode="no",
            mu=0,
            sigma=1,
            noise_flag=False
        ):
            super().__init__()
            # self.lr = lr
            self.exp_name = exp_name
            self.knowledge_flag = knowledge_flag
            self.run_mode = run_mode
            self.actions = [">", "<", "="]
            self.max_predict = (match_max_size + 1) * (len(eff_cols) + 1)
            self.events = np.loadtxt(events_path, dtype='str')
            self.num_events = len(self.events)
            self.match_max_size = match_max_size
            self.max_values = max_values
            self.window_size = window_size
            self.mu = mu
            self.sigma = sigma
            self.normailze_values = normailze_values
            self.embedding_events = nn.Embedding(num_events + 1, 3)
            self.embedding_values = [nn.Embedding(max_val, 3) for max_val in max_values]
            self.pattern_path = pattern_path.split("/")[-1].split(".")[0]
            self.noise_flag = noise_flag
            if init_flag:
                if not os.path.exists(f"Processed_Data/{self.exp_name}/{self.window_size}.pt"):
                    self.data = self._create_data(data_path)
                    self.data = self.data.view(len(self.data), -1)
                    self.data = self.data.detach().clone().requires_grad_(True)
                    torch.save(self.data, f"Processed_Data/{self.exp_name}/{self.window_size}.pt")
                else:
                    self.data = torch.load(f"Processed_Data/{exp_name}/{self.window_size}.pt").requires_grad_(True)
            global EMBDEDDING_TOTAL_SIZE
            EMBDEDDING_TOTAL_SIZE = 8
            self.hidden_size1 = hidden_size1
            self.hidden_size2 = hidden_size2
            self.num_cols = len(eff_cols)
            # self.num_actions = (len(self.actions) * (self.match_max_size + 1)) * 2 * 2 + 1  # [>|<|= * [match_max_size + 1(value)] * not / reg] * (and/or) |nop
            self.num_actions = len(self.actions) * 2 * 2
            # (num_events + 1) * (Sum_{i=0}^{i=num_cols} {(num_actions)^i})
            self.num_actions_in_one_event = sum([self.num_actions ** i for i in range(0, self.num_cols + 1)])
            # self.num_actions_in_use = (self.num_events) * self.num_actions_in_one_event + 1
            self.num_actions_in_use = (self.num_events) * self.num_actions_in_one_event
            print(self.num_actions_in_one_event, self.num_actions_in_use)

            self.embedding_actions = nn.Embedding(self.num_actions + 1, 1)
            self.embedding_desicions = nn.Embedding(self.num_events + 1, 1)
            self.actor_critic = ActorCriticModel(
                                num_events=self.num_events,
                                match_max_size=self.match_max_size,
                                window_size=self.window_size,
                                num_cols=self.num_cols,
                                hidden_size1=self.hidden_size1,
                                hidden_size2=self.hidden_size2,
                                embeddding_total_size=EMBDEDDING_TOTAL_SIZE,
                                num_actions=self.num_actions_in_use
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
            self.certainty = 0
            if not pattern_path == "" and self.knowledge_flag:
                self.knn = self._create_df(pattern_path)
            self.max_time = max_time
            self.count = 0
            self.min_values_bayes = [-i for i in normailze_values]
            self.max_values_bayes = [i - j for i,j in zip(max_values, normailze_values)]
            self.count_events = 1
            self.count_comparisons = 1

            self.action_counter = np.ones(self.num_actions_in_use)

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
            if not os.path.exists(f"Processed_knn/{self.pattern_path}"):
                print("Creating Knn!")
                os.mkdir(f"Processed_knn/{self.pattern_path}")
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


            knn = KNN(n_neighbors=2)
            knn.fit(df_new, df["rating"])
            self.knn_avg = df.rating.mean()

            test_pred = ratingPredictor(df_new, df["rating"], noise_flag=self.noise_flag, mu=self.mu, sigma=self.sigma)
            self.pred_optim = torch.optim.Adam(params=test_pred.parameters(), lr=3e-5)
            self.pred_sched = StepLR(self.pred_optim, step_size=2000, gamma=0.85)
            test_pred.df_knn_rating = []

            # TODO: add conditions to load no/semi/full knowledge!
            # TODO: must un-comment!

            if self.run_mode == "no":
                self.certainty = test_pred._train(self.pred_optim, self.pred_sched, count=0, max_count=0, max_total_count=0, n=0)
                test_pred.num_examples_given = 0
            elif self.run_mode == "semi":
                # test_pred._train(self.pred_optim, self.pred_sched, count=0, max_count=7, max_total_count=100, n=0)

                if not os.path.exists(f"Processed_knn/{self.pattern_path}/rating_model.pt"):
                    # pass
                    test_pred._train(self.pred_optim, self.pred_sched, count=0, max_count=10, max_total_count=100, n=10)
                    # torch.save(test_pred, f"Processed_knn/{self.pattern_path}/rating_model.pt")
                else:
                    print("Loaded pattern rating model! \n")
                    # test_pred = torch.load(f"Processed_knn/{self.pattern_path}/rating_model.pt")
                test_pred.num_examples_given = 3500
                self.certainty = test_pred._train(self.pred_optim, self.pred_sched, count=0, max_count=0, max_total_count=0, n=0)

            else: #self.run_mode == "full"
                # full knowledge run!
                self.certainty = 1

            test_pred.df_knn_rating = []

            self.pred_pattern = test_pred
            self.pred_pattern.rating_df_unlabeld = None
            self.pred_pattern.unlabeld_strs = []
            return knn

        def _create_data(self, data_path):
            date_time_obj = None
            all_data = None
            if self.exp_name == "Football":
                data = None
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
                            for i, value in enumerate(values[:len(self.normailze_values)]):
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
                for i in range(0, len(data) - self.window_size):
                    if i % 1000 == 0:
                        print(i)
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
            if not os.path.exists(f"Model/training/{self.exp_name}"):
                os.mkdir(f"Model/training/{self.exp_name}")
            lines = []
            if self.exp_name == "Football":
                with open(data_path) as f:
                    for line in f:
                        lines.append(line)

                for i in range(0, len(lines) - self.window_size):
                    with open(f"Model/training/{self.exp_name}/{i}.txt", "w") as f:
                        for j in range(i, i + self.window_size):
                            f.write(lines[j])
            elif self.exp_name == "StarPilot":
                current_files_created = 0
                files = os.listdir(data_path)[:200]
                for file in files:
                    lines = []
                    with open(os.path.join(data_path, file)) as f:
                        for line in f:
                            lines.append(line)

                        for i in range(0, len(lines) - self.window_size):
                            with open(f"Model/training/{self.exp_name}/{str(current_files_created)}.txt", "w") as new_file:
                                for j in range(i, i + self.window_size):
                                    new_file.write(lines[j])
                            current_files_created += 1
            else:
                raise Exception("Data set not supported")

        def forward(self, data, training_factor=0.0):

            base_value, event_after_softmax = self.actor_critic.forward_actor(data, training_factor=training_factor)
            value_reward, value_rating = self.actor_critic.forward_critic(base_value)
            return event_after_softmax, value_reward, value_rating


        def get_action(self, data, index=0, training_factor=0.0):
            global total_steps_trained
            total_steps_trained += 1
            probs, value_reward, value_rating = self.forward(Variable(data), training_factor=training_factor)
            numpy_probs = probs.detach().cpu().numpy()

            action = None

            numpy_probs = np.squeeze(numpy_probs).astype(float)

            z = np.copy(numpy_probs)
            ucb_factor = np.array([sqrt((2 * log(self.count_events))/ (self.action_counter[i])) for i, _ in enumerate(numpy_probs)])
            if index > 1:
                ucb_factor = ucb_factor / np.sum(ucb_factor)
                numpy_probs += ucb_factor

            numpy_probs = numpy_probs / np.sum(numpy_probs)
            try:

                action = np.random.multinomial(
                    n=1, pvals=numpy_probs, size=1
                )
                action = np.argmax(action)

                self.count_events += 1
                self.action_counter[action] += 1
            except Exception as e:
                print(e)
                raise e

            entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2

            log_prob = torch.log(probs.squeeze(0)[action])
            if abs(log_prob) < 0.1:
                self.count += 1

            #TODO: return all mini-actions and split event from action
            return action, log_prob, value_reward, value_rating, entropy



    def update_policy_batch(policy_network, ratings, rewards, log_probs, values_rating, values_reward,
                    Qval_rating, Qval_reward, entropy_term, epoch_idx, flag=False, certainty=1.0, run_type=False):

        def l1_penalty(log_probs, l1_lambda=0.001):
            """
            Returns the L1 penalty of the params.
            """
            l1_norm = sum(log_prob.abs().sum() for log_prob in log_probs)
            return l1_lambda*l1_norm / len(log_probs)

        actor_loss = None
        critic_loss = None
        for i in range(len(ratings)):
            Qvals_reward = np.zeros_like(values_reward[i])
            Qvals_rating = np.zeros_like(values_rating[i])
            curr_rewards = rewards[i]
            curr_ratings = ratings[i]
            curr_Qval_reward = Qval_reward[i]
            curr_Qval_rating = Qval_rating[i]
            curr_logs = log_probs[i]
            for t in reversed(range(len(curr_rewards))):
                Qval_0 = curr_rewards[t] + GAMMA * curr_Qval_reward
                Qval_1 = curr_ratings[t] * certainty + GAMMA * curr_Qval_rating
                Qvals_reward[t] = Qval_0
                Qvals_rating[t] = Qval_1

            if not run_type: #(Normal model)
                values = torch.FloatTensor((values_reward[i], values_rating[i])).requires_grad_(True)
                Qvals = torch.FloatTensor((Qvals_reward, Qvals_rating)).requires_grad_(True)
            else: #(Gain knowledge model)
                values = torch.FloatTensor(values_rating[i]).requires_grad_(True)
                Qvals = torch.FloatTensor(Qvals_rating).requires_grad_(True)

            # print(curr_logs)
            curr_logs = torch.stack(curr_logs).requires_grad_(True)
            advantage = Qvals - values
            advantage = advantage.to(curr_logs .device)
            log_probs_reg = l1_penalty(curr_logs , l1_lambda=0.05)

            # print(curr_logs)
            curr_actor_loss = (-curr_logs  * advantage).mean().requires_grad_(True)

            curr_actor_loss += log_probs_reg
            curr_critic_loss = 0.5 * advantage.pow(2).mean().requires_grad_(True)

            if actor_loss is None:
                actor_loss = curr_actor_loss
                critic_loss = curr_critic_loss
            else:
                # print(curr_actor_loss)
                actor_loss += curr_actor_loss
                critic_loss += curr_critic_loss

        # / 2 only to compare to test lower mini batch size, from 14.9.21 should be divided by mini-batch size!
        policy_network.actor_optimizer.zero_grad()
        policy_network.critic_optimizer.zero_grad()
        # actor_loss_1 = actor_loss.cpu().detach().numpy() / 2
        actor_loss_1 = actor_loss.cpu().detach().numpy() / len(ratings)
        actor_loss = actor_loss.cuda()
        # critic_loss_1 = critic_loss.cpu().detach().numpy() / 2
        critic_loss_1 = critic_loss.cpu().detach().numpy() / len(ratings)
        critic_loss = critic_loss.cuda()
        if flag:
            actor_loss.backward()
            policy_network.actor_optimizer.step()
        else:
            critic_loss.backward()
            policy_network.critic_optimizer.step()

        return actor_loss_1, critic_loss_1



    def train(model, num_epochs=5, test_epcohs=False, split_factor=0, bs=0, mini_batch_size=16, rating_flag=True, run_name=None, pretrain_flag=False, wandb_name=""):
        actor_loss, critic_loss =  None, None
        new_run_name = wandb_name
        run_type = (run_name == 'gain_knowledge_model')
        pred_flag = (model.run_mode != "full")
        if run_name is None:
            run_name = new_run_name
        else:
            run_name = new_run_name + "_" + run_name
        not_finished_count = 0

        run = wandb.init(project='Pattern_Mining_baseline', entity='guyshapira', name=run_name, settings=wandb.Settings(start_method='fork'))
        config = wandb.config
        config.hidden_size1 = model.hidden_size1
        config.hidden_size2 = model.hidden_size2
        config.current_epoch = 0
        config.batch_size = bs
        config.window_size = model.window_size
        config.num_epochs = num_epochs
        config.split_factor = split_factor
        config.total_number_of_steps = total_steps_trained
        config.name = wandb_name
        if model.noise_flag:
            config.mu = model.mu
            config.sigma = model.sigma

        added_info_size = (model.match_max_size + 1) * (model.num_cols + 1)

        total_best = -1
        best_found = {}
        actuall_best_found = {}
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
        entropy_term =  0

        training_factor = 0.8
        pbar_file = sys.stdout
        total_count = 0
        count_actor = 0
        count_critic = 0

        recent_ratings, recent_rewards, recent_logs = [], [], []
        recent_values_ratings, recent_values_rewards = [], []
        recent_Qval_ratings, recent_Qval_rewards = [], []

        model.count = 0

        global num_epochs_trained
        for epoch in range(num_epochs):
            config.update({"current_epoch" : epoch}, allow_val_change=True)

            if num_epochs_trained is None:
                num_epochs_trained = 0
            else:
                num_epochs_trained += 1
            currentPath = pathlib.Path(os.path.dirname(__file__))
            absolutePath = str(currentPath.parent)
            sys.path.append(absolutePath)

            with tqdm.tqdm(total=bs, file=pbar_file) as pbar:
                in_round_count = 0
                path = os.path.join(absolutePath, "Model", "training", model.exp_name)
                data_len = len(os.listdir(path))
                for index in range(epoch + 2, min(data_len - 2, len(model.data)), data_len // bs):
                    set_data = None
                    if in_round_count >= bs:
                        break
                    total_count += 1
                    in_round_count += 1
                    step_list = None
                    if pretrain_flag:
                        step_list = [10, 25, 40, 60]
                    else:
                        step_list = [25, 100, 250, 500]
                    if in_round_count in step_list and not model.run_mode == "full":
                        model.pred_pattern.save_all()
                        model.pred_pattern.train()
                        if pretrain_flag:
                            model.pred_optim = torch.optim.Adam(params=model.pred_pattern.parameters(), lr=1e-4)
                            model.certainty = model.pred_pattern._train(model.pred_optim, None, count=0, max_count=3, max_total_count=50, n=175, retrain=True)

                        else:
                            model.pred_optim = torch.optim.Adam(params=model.pred_pattern.parameters(), lr=7e-5)
                            n = 75
                            if model.run_mode == "semi":
                                n = 50
                            model.certainty = model.pred_pattern._train(model.pred_optim, None, count=0, max_count=2, max_total_count=50, n=n, retrain=True)
                    data = model.data[index]

                    data_size = len(data)
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
                    patterns, special_reward = [] , []
                    normalize_rating, normalize_reward = [], []
                    if total_count % 75 == 0:
                        training_factor /= 1.2
                    while not is_done:
                        if not set_data is None:
                            data = set_data.clone().detach().requires_grad_(True)
                            set_data = None
                        data = data.cuda()

                        action, log_prob, value_reward, value_rating, entropy = model.get_action(
                            data, in_round_count, training_factor=training_factor
                        )
                        count += 1
                        value_rating = value_rating.detach().cpu().numpy()[0]
                        value_reward = value_reward.detach().cpu().numpy()[0]
                        values_rating.append(value_rating)
                        values_reward.append(value_reward)
                        entropy_term += entropy

                        if action == model.num_actions_in_use:
                            ratings.append(1)
                            log_probs.append(log_prob)
                            if len(actions) == 0:
                                special_reward.append(-1.5)
                                break
                            else:
                                special_reward.append(10)
                                break
                        else:
                            special_reward.append(None)
                            event, mini_actions, conds, comps_to, comp_vals = mapping_for_baseline(action, model.events, count, model.num_actions_in_one_event, model.num_cols ,model.actions)
                            # print(event)
                            # print(mini_actions)
                            # print(conds)
                            # print(comps_to)
                            events.append(event)
                            all_comps.append(comps_to)
                            entropy_term += entropy
                            log_probs.append(log_prob)
                            actions.append(mini_actions)

                            file = os.path.join(absolutePath, "Model", "training", model.exp_name, "{}.txt".format(index))

                            all_conds.append(conds)
                            if comp_vals.count("nop") != len(comp_vals):
                                bayesian_dict = set_values_bayesian(comp_vals,
                                    model.all_cols, model.cols, mini_actions, event,
                                    all_conds, file, model.max_values_bayes,
                                    model.min_values_bayes
                                )
                                store_to_file(events, actions, index, comp_values, model.cols, all_conds, comp_vals, all_comps, model.max_fine_app, model.max_time, model.exp_name)
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
                            pattern = OpenCEP_pattern(
                                model.exp_name, events, actions, index, comp_values,
                                    model.cols, all_conds, all_comps, model.max_time
                            )
                            patterns.append(pattern)
                            str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols, all_comps)

                            rating, norm_rating = rating_main(model, events, all_conds, actions, str_pattern, rating_flag, epoch, pred_flag=pred_flag, noise_flag=model.noise_flag)

                            ratings.append(rating)
                            normalize_rating.append(norm_rating)

                        if count >= model.match_max_size:
                            is_done = True


                    # after loop ended- calc reward for all patterns and update policy
                    try:
                        run_exp_name = model.exp_name
                        run_OpenCEP(exp_name=run_exp_name, test_name=index, patterns=patterns)
                    except Exception as e:
                        # raise(e)
                        # timeout error
                        finished_flag = False
                        not_finished_count += 1

                    content = None
                    try:
                        with open("Data/Matches/{}Matches.txt".format(index), "r") as f:
                            content = f.read()
                    except Exception as e:
                        print(e)
                        # Probably becasue of not finished pattern case
                        content = None

                    for pattern_index, pattern_rating in enumerate(ratings):
                        sp_rew = special_reward[pattern_index]
                        if not sp_rew is None:
                            rewards.append(sp_rew)
                            real_rewards.append(sp_rew)
                            normalize_reward.append(sp_rew - 20)

                        else:
                            if not finished_flag:
                                reward = -5
                                rewards.append(reward)
                                real_rewards.append(reward)
                                normalize_reward.append(reward - 20)
                                is_done = True
                            else:
                                original_reward = int(content.count(f"{pattern_index}: "))
                                reward = original_reward
                                if reward >= model.max_fine_app:
                                    reward = max(0, 2 * model.max_fine_app - reward)
                                real_rewards.append(reward)
                                normalize_reward.append(reward - 20)

                    try:
                        if finished_flag:
                            near_windows_rewards = calc_near_windows(model.exp_name, index, patterns,
                                model.max_fine_app, data_len)

                            for pattern_index, pattern_rating in enumerate(ratings):
                                sp_rew = special_reward[pattern_index]
                                if sp_rew is None:
                                    reward = real_rewards[pattern_index] * 0.75 + near_windows_rewards[pattern_index] * 0.25
                                    actuall_rating, _ = rating_main(model, events, all_conds, actions, str_pattern, rating_flag, epoch, pred_flag=False, noise_flag=model.noise_flag)
                                    actuall_best_found_reward = reward * actuall_rating

                                    reward *= pattern_rating

                                    rewards.append(reward)

                                if len(best_found) < 10:
                                    best_found.update({reward: pattern})
                                    actuall_best_found.update({reward: actuall_best_found_reward})
                                else:
                                    worst_reward = sorted(list(best_found.keys()))[0]
                                    if reward > worst_reward:
                                        del best_found[worst_reward]
                                        del actuall_best_found[worst_reward]
                                        best_found.update({reward: pattern})
                                        actuall_best_found.update({reward: actuall_best_found_reward})


                    except Exception as e:
                        print(e)
                        # raise e
                        if not finished_flag:
                            continue
                        else:
                            rewards = [pattern_rating * real_rew for pattern_rating, real_rew in zip(ratings, real_rewards)]



                    _, Qval_reward, Qval_rating = model.forward(data, training_factor=training_factor)
                    Qval_rating = Qval_rating.detach().cpu().numpy()[0]
                    Qval_reward = Qval_reward.detach().cpu().numpy()[0]

                    del data
                    gc.collect()
                    actor_flag = False

                    if count_actor < 600:
                        actor_flag = True
                        count_actor += len(ratings)
                    elif count_critic < 1500:
                        count_critic += len(ratings)
                    else:
                        count_actor = 0
                        count_critic = 0


                    index_max = np.argmax(rewards)
                    # if total_steps_trained > 4500 and real_rewards[index_max] <= 0:
                    #     continue


                    # else:
                    recent_ratings.append(ratings)
                    recent_rewards.append(real_rewards)
                    recent_logs.append(log_probs)
                    recent_values_ratings.append(values_rating)
                    recent_values_rewards.append(values_reward)
                    recent_Qval_ratings.append(Qval_rating)
                    recent_Qval_rewards.append(Qval_reward)

                    if len(recent_ratings) == mini_batch_size:
                        actor_loss , critic_loss  = update_policy_batch(model, recent_ratings, recent_rewards, recent_logs, recent_values_ratings, recent_values_rewards,
                                                recent_Qval_ratings, recent_Qval_rewards,
                                                entropy_term, epoch, flag=actor_flag, certainty=model.certainty,
                                                run_type=run_type)
                        print(f"Updated! actor loss {actor_loss}")
                        recent_ratings, recent_rewards, recent_logs = [], [], []
                        recent_values_ratings, recent_values_rewards = [], []
                        recent_Qval_ratings, recent_Qval_rewards = [], []

                    all_ratings.append(np.sum(ratings))
                    all_rewards.append(rewards[index_max])
                    numsteps.append(len(actions))
                    avg_numsteps.append(np.mean(numsteps))
                    mean_rewards.append(np.mean(all_rewards))
                    max_rating.append(np.max(ratings))
                    real.append(real_rewards[index_max])
                    rating_plot.append(ratings[index_max])
                    mean_real.append(np.mean(real_rewards))

                    # if in_round_count % 2 == 0:
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
                            num_examples_given = model.pred_pattern.get_num_examples()
                            if not actor_loss is None:
                                wandb.log({"reward": real_rewards[index_max], "rating": ratings[index_max],
                                        "max rating": np.max(ratings), "actor_flag": int(actor_flag),
                                        "actor_loss_reward": actor_loss, "critic_loss_reward": critic_loss,
                                        "curent_step": total_steps_trained,
                                        "certainty": model.certainty,
                                        "num_examples": num_examples_given,
                                        "training_factor": training_factor})
                            else:
                                wandb.log({"reward": real_rewards[index_max], "rating": ratings[index_max],
                                        "max rating": np.max(ratings), "actor_flag": int(actor_flag),
                                        "curent_step": total_steps_trained,
                                        "certainty": model.certainty,
                                        "num_examples": num_examples_given,
                                        "training_factor": training_factor})

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
                        sys.stdout.write(f"\n--- Current count {model.count} ---\n")

                    config.update({"total_number_of_steps" : total_steps_trained}, allow_val_change=True)
                    #TODO: hyper-param, need to find proper value
                    if model.count > 50:
                        print("\n\n\n---- Stopping early because of low log ----\n\n\n")
                        model.count = 0
                        for g1, g2 in zip(model.actor_optimizer.param_groups, model.critic_optimizer.param_groups):
                            g1['lr'] *= 0.85
                            g2['lr'] *= 0.85
                        # continue
                        training_factor = 0.6
                        # break



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
                # for rew, rat, max_rat in zip(real_groups[-int(bs / GRAPH_VALUE):], rating_groups[-int(bs / GRAPH_VALUE):], max_ratings_group[-int(bs / GRAPH_VALUE):]):
                #     wandb.log({"reward": rew, "rating": rat, "max rating": max_rat})

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

                if False:
                    after_epoch_test(best_pattern)
                    with open("Data/Matches/allMatches.txt", "r") as f:
                        results.append(int(f.read().count("\n") / (max_len_best + 1)))
                    os.remove("Data/Matches/allMatches.txt")


            mean_result_out, out_sample_acc  = run_test(model, load_flag=False, avg_score=np.mean(real), rating_flag=rating_flag, pred_flag=pred_flag)
            wandb.log({"out_sample_acc": out_sample_acc, "out_sample_mean_reward": mean_result_out,
                    "test_accuracy": model.certainty, "mean_result_over_best_patterns": np.mean(list(best_found.keys()))})
            if not model.run_mode == "full":
                wandb.log({"number_of_examples": model.pred_pattern.get_num_examples(),
                "actuall_mean_result_over_best": np.mean(list(actuall_best_found.values()))})
        cuda_handle.empty_cache()
        best_res = - 10
        for dict_res in factor_results:
            new_res = dict_res['rating'] / 10 + dict_res['reward'] / model.max_fine_app
            if new_res > best_res:
                best_res = new_res
        # return best_res


        timestr = time.strftime("%Y%m%d-%H%M%S")
        prefix_path = "Model/Weights"
        torch.save(model.state_dict(), prefix_path + "/Model/" + timestr + ".pt")
        torch.save(model.pred_pattern.state_dict(), prefix_path + "/Pattern/" + timestr + ".pt")


        out_sample_acc, mean_result_out = run_test(model, load_flag=False, avg_score=np.mean(real), rating_flag=rating_flag, pred_flag=pred_flag)

        if not run_type:
            wandb.run.summary["test_accuracy"] = model.certainty
            wandb.run.summary["mean_result_over_best_patterns"] = np.mean(list(best_found.keys()))
            wandb.run.summary["out_of_sample_acc"] = out_sample_acc
            wandb.run.summary["out_of_sample_mean_reward"] = mean_result_out

            if model.run_mode == "full":
                wandb.run.summary["number_of_examples"] = total_steps_trained
                wandb.run.summary["actuall_mean_result_over_best"] = np.mean(list(best_found.keys()))
            else:
                wandb.run.summary["number_of_examples"] = model.pred_pattern.get_num_examples()
                wandb.run.summary["actuall_mean_result_over_best"] = np.mean(list(actuall_best_found.values()))

        run.finish()
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


    def run_test(model, name="", load_flag=False, avg_score=0.0, rating_flag=False, pred_flag=False):
        #TODO: convert to pattern form!
        prefix_path = "Model/Weights"
        if load_flag:
            # model.load(prefix_path + "/Model/" + name + ".pt")
            model.pred_pattern.load_state_dict(torch.load(prefix_path + "/Pattern/" + name + ".pt"))
        model.eval()
        if model.exp_name == "StarPilot":
            df = pd.read_csv("Patterns/test_StarPilot.csv")[["rating", "events", "conds", "actions", "pattern_str"]]
        elif model.exp_name == "Football":
            df = pd.read_csv("Patterns/test_Football.csv")[["rating", "events", "conds", "actions", "pattern_str"]]
        else:
            raise Exception("Data set not supported!")

        df.rating = df.rating.apply(lambda x : min(round(float(x) - 1), 49))
        diff = 0.0
        sum_out_of_sample = 0.0

        for _, row in df.iterrows():

            events = ast.literal_eval(row['events'])
            real_rating = row['rating']
            str_pattern = row['pattern_str']
            if isinstance(str_pattern, float):
                str_pattern = ""
            actions = ast.literal_eval(row['actions'])
            all_conds = []
            rating, _ = rating_main(model, events, all_conds, actions, str_pattern, rating_flag=rating_flag, pred_flag=pred_flag, flat_flag=True)
            diff_val = rating - int(real_rating)
            add_value = 1.0
            if diff_val <= 7:
                add_value = 0.25
            elif diff_val <= 15:
                add_value = 0.5

            if diff_val >= 1:
                diff += add_value
            sum_out_of_sample += rating * avg_score

        print(f"mean new pattern = {sum_out_of_sample / len(df)}")
        print(f"Acc = {1 - diff / len(df)}")
        return sum_out_of_sample / len(df), (1 - (diff / len(df)))



    def main(parser):
        args = parser.parse_args()
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        max_vals = [int(i) for i in args.max_vals.split(",")]
        norm_vals = [int(i) for i in args.norm_vals.split(",")]
        all_cols = args.all_cols.replace(" ", "").split(",")
        eff_cols = args.eff_cols.replace(" ", "").split(",")
        if args.pattern_path == "":
            rating_flag = False
        else:
            rating_flag = True

        results = {}
        suggested_models = []
        all_patterns = []
        global class_inst
        class_inst = ruleMiningClass(data_path=args.data_path,
                                    pattern_path=args.pattern_path,
                                    events_path=args.events_path,
                                    num_events=args.num_events,
                                    match_max_size=args.max_size,
                                    window_size=args.window_size,
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
                                    init_flag=True,
                                    knowledge_flag=args.early_knowledge,
                                    run_mode=args.run_mode,
                                    mu=args.mu,
                                    sigma=args.sigma,
                                    noise_flag=args.noise_flag)

        print("Finished creating Training model")

        if not args.early_knowledge: # pre-training is needed
            pretrain_inst = ruleMiningClass(data_path=args.data_path,
                pattern_path=args.pattern_path,
                events_path=args.events_path,
                num_events=args.num_events,
                match_max_size=args.max_size,
                window_size=args.window_size,
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
                init_flag=True,
                run_mode=args.run_mode,
                mu=args.mu,
                sigma=args.sigma,
                noise_flag=args.noise_flag)
            print("Finished creating Knowledge model")

            train(pretrain_inst, num_epochs=4, bs=75, mini_batch_size=args.mbs, split_factor=0.5, rating_flag=True, run_name="gain_knowledge_model", pretrain_flag=True, wandb_name=args.wandb_name)
            #copy rating model to trainable model

            #TODO: check if there is a problem in here?
            #TODO: double pre training, do this twich and see if this achievs something
            class_inst.certainty = pretrain_inst.certainty
            class_inst.pred_pattern = pretrain_inst.pred_pattern
            class_inst.knn = pretrain_inst.knn
            class_inst.list_of_dfs = pretrain_inst.list_of_dfs
        # train working model
        result, patterns = train(class_inst, num_epochs=args.epochs, bs=args.bs, mini_batch_size=args.mbs, split_factor=args.split_factor, rating_flag=rating_flag, run_name="train_model", pretrain_flag=False, wandb_name=args.wandb_name)

        all_patterns.append(patterns)
        cuda_handle.empty_cache()
        # print(patterns)
        # results.update({split_factor: result})
        # suggested_models.append({split_factor: class_inst})


        # if 0:
        #     print(results)
        #     pareto_results = np.array(list(results.values()))
        #     pareto_results = np.array([np.array(list(res.values())) for res in pareto_results])
        #     print(pareto_results)
        #     patero_results = is_pareto_efficient(pareto_results)
        #     good_patterns = []
        #     for patero_res, model, patterns in zip(patero_results, suggested_models, all_patterns):
        #         if patero_res:
        #             print(model)
        #             good_patterns.extend(list(patterns))
        #             print(patterns)


        #     run_OpenCEP(events=args.final_data_path, patterns=good_patterns, test_name="secondLevel27April")

        return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CEP pattern miner')
    parser.add_argument('--bs', default=500, type=int, help='batch size')
    parser.add_argument('--mbs', default=24, type=int, help='mini batch size')
    parser.add_argument('--epochs', default=5, type=int, help='num epochs to train')
    parser.add_argument('--lr_actor', default=3e-5, type=float, help='starting leartning rate for actor')
    parser.add_argument('--lr_critic', default=5e-4, type=float, help='starting learning rate for critic')
    parser.add_argument('--hidden_size1', default=1024, type=int, help='hidden_size param for model')
    parser.add_argument('--hidden_size2', default=2048, type=int, help='hidden_size param for model')
    parser.add_argument('--max_size', default=8, type=int, help='max size of pattern')
    parser.add_argument('--max_fine_app', default=80, type=int, help='max appearance of pattnern in a single window')
    parser.add_argument('--pattern_max_time', default=100, type=int, help='maximum time for pattern (seconds)')
    parser.add_argument('--window_size', default=485, type=int, help='max size of input window')
    parser.add_argument('--num_events', default=41, type=int, help='number of unique events in data')
    parser.add_argument('--split_factor', default=0.2, type=float, help='split how much train to rating and how much for reward')
    parser.add_argument('--data_path', default='StarPilot/GamesExp/', help='path to data log')

    parser.add_argument('--events_path', default='StarPilot/EventsExp', help='path to list of events')


    parser.add_argument('--pattern_path', default='Patterns/pattern28_50_ratings.csv', help='path to known patterns')
    parser.add_argument('--final_data_path', default='store_folder/xaa', help='path to next level data')
    parser.add_argument('--max_vals', default = "50, 50, 50, 50, 5", type=str, help="max values in columns")
    parser.add_argument('--norm_vals', default = "0, 0, 0, 0, 0", type=str, help="normalization values in columns")
    parser.add_argument('--all_cols', default = 'x, y, vx, vy, health', type=str, help="all cols in data")
    parser.add_argument('--eff_cols', default = 'x, y, vx', type=str, help="cols to use in model")
    parser.add_argument('--early_knowledge', dest='early_knowledge',
                type=lambda x: bool(strtobool(x)),
                default = True, help="indication if expert knowledge is available")

    parser.add_argument('--noise_flag', dest='noise_flag',
                type=lambda x: bool(strtobool(x)),
                default = False, help="indication if expert values has noise in them")
    parser.add_argument('--sigma', default=1, type=float, help='sigma for gaussian distribution')
    parser.add_argument('--mu', default=0, type=float, help='mu for gaussian distribution')
    parser.add_argument('--seed', default=0, type=int, help='seed for all random libraries')
    parser.add_argument('--wandb_name', default = 'full_knowledge', type=str)
    parser.add_argument('--exp_name', default = 'StarPilot', type=str)
    parser.add_argument('--run_mode', default="no", type=str, choices=['no', 'semi', 'full'], help="run mode, semi for cool name model, \n"\
                                                                                                "full for supervised baseline, \n"\
                                                                                                "no for unsuervised baseline \n")

    torch.set_num_threads(80)
    main(parser)
