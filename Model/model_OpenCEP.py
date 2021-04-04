import torch
import random
import torch.nn as nn
import os
from Model.utils import (
    to_var,
    mapping,
    OpenCEP_pattern,
    pattern_complexity,
    after_epoch_test,
    new_mapping,
    get_action_type,
    create_pattern_str,
    store_patterns_and_rating_to_csv,
    set_values,
    ball_patterns,
    bayesian_function,
    set_values_bayesian,
    set_values_bayesian2,
    store_to_file,
    replace_values,
)
import tqdm
import pathlib
from bayes_opt import BayesianOptimization
import ast
import sys
import time
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from shutil import copyfile
import datetime
from difflib import SequenceMatcher
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
GRAPH_VALUE = 50
GAMMA = 0.99
EMBDEDDING_TOTAL_SIZE = 21
PAD_VALUE = -5.5
device = "cuda:6"
# EMBDEDDING_TOTAL_SIZE = 12


class ruleMiningClass(nn.Module):
    # TODO: switch all max_inputs to lists of len num_cols
    def __init__(
        self,
        data_path,
        num_events,
        match_max_size=8,
        max_values=None,
        normailze_values=None,
        window_size=350,
        max_count=2000,
        num_cols=5,
        max_fine_app=55,
    ):
        super().__init__()
        self.actions = [">", "<", "=", "+>", "->", "*="]
        # self.actions = [">", "<", "="]
        self.num_events = num_events
        self.match_max_size = match_max_size
        self.max_values = max_values
        self.window_size = window_size
        self.normailze_values = normailze_values
        self.embedding_events = nn.Embedding(num_events + 1, 3)
        self.embedding_values = [nn.Embedding(max_val, 3) for max_val in max_values]
        self.embedding_count = nn.Embedding(max_count, 3)
        self.data = self._create_data(data_path)
        self.data = self.data.view(len(self.data), -1)
        self.hidden_size = 4096
        self.num_cols = num_cols
        self.num_actions = (len(self.actions) * (self.match_max_size + 1)) * 2 * 2 + 1  # [>|<|= * [match_max_size + 1(value)] * not / reg] * (and/or) |nop
        self.embedding_actions = nn.Embedding(self.num_actions + 1, 1)
        self.embedding_desicions = nn.Embedding(self.num_events + 1, 1)
        self.linear_base = nn.Linear(
            self.window_size * EMBDEDDING_TOTAL_SIZE + (self.match_max_size + 1)  * (self.num_cols + 1),
            self.hidden_size,
        ).cuda()
        self.event_tagger = nn.Linear(self.hidden_size, self.num_events + 1).cuda()

        self.action_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.num_actions)
                for _ in range(self.num_cols)
            ]
        ).cuda()
        self.critic = nn.Linear(self.hidden_size, 1).cuda()

        self._create_training_dir(data_path)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        self.all_cols = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
        self.cols  = ["x", "y", "z", "vx", "vy"]
        self.max_fine_app = max_fine_app
        self.knn_avg = 0
        self.knn = self._create_df()
        self.count = 0
        self.min_values_bayes = [-i for i in normailze_values]
        self.max_values_bayes = [i - j for i,j in zip(max_values, normailze_values)]

    def _create_df(self):
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        def custom_dist(x, y):
            print(x)
            print(y)
            input()
            # print(y)
            # print(x)
            # print(type(x))
            # print(len(x))
            return 3
            sim = similar(x, y)
            return sim

        def fix_str_list_columns(data, flag=False):
            data = data[1:]
            data = data[:-1]
            data = data.replace("\"", "")
            data = data.replace("\'", "")
            data = data.replace(" ", "")
            temp = pd.Series(data)
            temp = temp.str.split(",", expand=True)
            return temp


        def fix_str_list_columns2(temp):
            for col in temp.columns:
                temp[col] = temp[col].astype('category')
                temp[col] = temp[col].cat.codes
            # combined = temp[temp.columns[1:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
            return temp

        def fix_int_list_columns(data):
            data = data[1:]
            data = data[:-1]
            return data.split(",")
            str_value = ''.join(data)
            str_value = str_value.replace(",", "")
            return [float(x) for x in str_value]


        self.list_of_dfs = []
        df = pd.read_csv("Patterns/pattern14.csv")[["rating", "events", "conds", "actions"]]
        # df = df[:800]

        df.rating = df.rating.apply(lambda x : round(float(x) + 0.5))
        str_list_columns = ["conds", "actions"]
        int_list_columns = ["events"]
        fit_columns = int_list_columns + str_list_columns
        df_new = None
        for col in fit_columns:
            temp = None

            for val in df[col]:
                if temp is None:
                    temp = fix_str_list_columns(val)
                else:
                    temp = temp.append(fix_str_list_columns(val))
            temp = temp.reset_index(drop=True)

            add_df = []
            for col_name in temp.columns:
                temp_dict = dict(zip(temp[col_name],temp[col_name].astype('category').cat.codes))
                temp_dict['Nan'] = -1
                add_df.append(temp_dict)
            self.list_of_dfs.append(add_df)
            combined = fix_str_list_columns2(temp)
            combined.columns = list(map(lambda x: col + "_" + str(x), combined.columns))

            if df_new is None:
                df_new = combined
            else:
                df_new = pd.concat([df_new, combined], axis=1).reset_index(drop=True)
            df_new = df_new.fillna(PAD_VALUE)
        print(df_new)
        # knn = KNN(n_neighbors=5, metric=custom_dist)
        knn = KNN(n_neighbors=1)
        knn.fit(df_new, df["rating"])
        self.knn_avg = df.rating.mean()
        return knn

    def _create_data(self, data_path):
        date_time_obj = None
        data = None
        with open(data_path) as f:
            for line in f:
                values = line.split("\n")[0]
                values = values.split(",")
                event = values[0]
                event = self.embedding_events(torch.tensor(int(new_mapping(event, reverse=True))))
                values = values[2:] # skip sid and ts
                embed_values = [self.embedding_values[i](torch.tensor(int(value) + self.normailze_values[i])) for (i,value) in enumerate(values)]
                embed_values.insert(0, event)
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

    def forward(self, input, mask, T=1):
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


        x = F.relu(self.linear_base(input.cuda()))

        value = self.critic(x)
        event_before_softmax = self.event_tagger(x)
        event_after_softmax = masked_softmax(event_before_softmax, mask.clone(), dim=0, T=T)
        # return x
        return event_after_softmax, value

    def get_event(self, input, index=0, mask=None, training_factor=0.0, T=1):
        probs, value = self.forward(Variable(input), mask, T=T)
        numpy_probs = probs.detach().cpu().numpy()
        if index % 10 == 0 and index != 0:
            print(numpy_probs)
            time.sleep(0)
            # exit()
        action = np.random.choice(
            self.num_events + 1, p=np.squeeze(numpy_probs)
        )


        entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2
        if np.random.rand() > 1 - training_factor:
            action = np.random.randint(len(mask))

        log_prob = torch.log(probs.squeeze(0)[action])
        if abs(log_prob) < 0.1:
            self.count += 1

        return action, log_prob, value, entropy


    def single_col_mini_action(self, data, index, training_factor=0.0):
        # replace get event
        x = F.relu(self.action_layers[index](data))
        probs = F.softmax(x, dim=0)
        numpy_probs = probs.detach().cpu().numpy()
        entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2

        if np.random.rand() > 1 - training_factor:
            highest_prob_action = np.random.randint(len(probs))
        else:
            highest_prob_action = np.random.choice(
                self.num_actions, p=np.squeeze(numpy_probs)
            )
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action]).cpu()

        highest_prob_value = None
        mini_action, _, _ = get_action_type(
            highest_prob_action, self.num_actions, self.actions, self.match_max_size
        )

        if len(mini_action.split("value")) > 1:
            highest_prob_value = "value"

        return highest_prob_action, highest_prob_value, log_prob, entropy

    def get_cols_mini_actions(self, data, training_factor=0.0):
        mini_actions = []
        log_probs = 0.0
        compl_vals = []
        conds = []
        mini_actions_vals = []
        total_entropy = 0
        comps_to = []
        updated_data = self.linear_base(data)
        for i in range(self.num_cols):
            # TODO: save return values and stuffx
            action, value, log, entropy = self.single_col_mini_action(updated_data, i, training_factor) #this is weird, should update data after actions
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

    # Qvals = (Qvals - Qvals.mean()) / (
    #             Qvals.std(unbiased=False) + 1e-9)
    # Qval -= Qvals.mean()
    # print(f"Qvals {Qvals}")
    # if flag:
    #     print(f"Qvals {Qvals}")
    #     print(f"values {values}")
    advantage = Qvals - values
    # advantage = (advantage - advantage.mean())
    # print(f"advantage {advantage}")

    actor_loss = (-log_probs * advantage).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()
    # print(f"Critic: {critic_loss}, Actor {actor_loss}\n")
    policy_gradient = actor_loss + critic_loss + 0.001 * entropy_term
    policy_gradient = policy_gradient.cuda()
    # print(policy_gradient)
    policy_network.optimizer.zero_grad()
    policy_gradient.backward(retain_graph=True)
    policy_network.optimizer.step()


def train(model, num_epochs=15, test_epcohs=False, round_number=75, temp_given=55):
    flatten = lambda list_list: [item for sublist in list_list for item in sublist]

    torch.autograd.set_detect_anomaly(True)
    added_info_size = (model.match_max_size + 1) * (model.num_cols + 1)
    added_info_size_knn = (model.match_max_size + 1) * (6 + 1)
    results = []
    total_best = -1
    all_rewards = []
    numsteps = []
    avg_numsteps = []
    mean_rewards = []
    real, mean_real = [], []
    rating_plot = []
    all_ratings = []
    entropy_term = 0
    turn_flag = 0
    training_factor = 0.50
    for epoch in range(num_epochs):
        if epoch % 3 == 0 and epoch > 1:
            for g in model.optimizer.param_groups:
                g['lr'] *= 0.5

        if epoch < 1:
            temper = 0.5
            print(temper)
        else:
            temper = 1

        pbar_file = sys.stdout
        num_batchs = 200
        with tqdm.tqdm(total=num_batchs, file=pbar_file) as pbar:
            in_round_count = 0
            for index in range(epoch, len(model.data), len(model.data) // num_batchs):
                in_round_count += 1
                if in_round_count % round_number == 0:
                    turn_flag = 1 - turn_flag
                data = model.data[index]
                data_size = len(data)
                old_desicions = torch.tensor([PAD_VALUE] * added_info_size)
                data2 = torch.cat((data, torch.tensor([PAD_VALUE] * added_info_size_knn).float()), dim=0)
                data = torch.cat((data, old_desicions.float()), dim=0)
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
                values = []
                comp_values = []
                patterns = []
                ratings = []
                mask = torch.tensor([1.0] * (model.num_events + 1))
                if in_round_count % 35 == 0 and epoch < 5:
                    temper /= 1.05
                if in_round_count % 25 == 0:
                    training_factor /= 1.1
                while not is_done:
                    data = data.cuda()
                    mask_orig = mask.clone()
                    action, log_prob, value, entropy = model.get_event(
                        data, in_round_count, mask, training_factor=training_factor, T=temper
                    )
                    data = data.clone()
                    data[data_size + count * (model.num_cols + 1)] = model.embedding_desicions(
                        torch.tensor(action)
                    ).cuda()
                    data2[data_size + count * (model.num_cols + 1)] = data[data_size + count * (model.num_cols + 1)]
                    count += 1
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
                            break
                    else:
                        event = new_mapping(action)
                        events.append(event)
                        mini_actions, log, comp_vals, conds, actions_vals, entropy, comps_to = model.get_cols_mini_actions(data, training_factor=training_factor)
                        all_comps.append(comps_to)
                        entropy_term += entropy
                        for j, action_val in enumerate(actions_vals):
                            data = data.clone()
                            try:
                                data[data_size + count * (model.num_cols + 1) + j + 1] = model.embedding_actions(
                                    torch.tensor(action_val))
                                data2[data_size + count * (model.num_cols + 1) + j + 1] = data[data_size + count * (model.num_cols + 1) + j + 1]
                            except Exception as e:
                                print(f"count {count}, j {j}")
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
                                model.all_cols, mini_actions, event,
                                all_conds, file, model.max_values_bayes,
                                 model.min_values_bayes
                             )
                            store_to_file(events, actions, index, comp_values, model.cols, all_conds, comp_vals, all_comps)
                            b_optimizer = BayesianOptimization(
                                f=bayesian_function,
                                pbounds=bayesian_dict,
                                random_state=42,
                                verbose=0,
                            )
                            try:
                                b_optimizer.maximize(
                                    init_points=5,
                                    n_iter=5,
                                )

                                selected_values = list(b_optimizer.max['params'].values())
                            except Exception as e:
                                print(bayesian_dict)
                                selected_values = [max(model.normailze_values) for _ in range(len(bayesian_dict))]
                            comp_vals = replace_values(comp_vals, selected_values)

                        comp_values.append(comp_vals)
                        pattern = OpenCEP_pattern(
                            events, actions, index, comp_values, model.cols, all_conds, all_comps
                        )
                        str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols, all_comps)
                        eff_pattern = pattern.condition
                        with open("Data/Matches/{}Matches.txt".format(index), "r") as f:
                            content = f.read()
                            reward = int(content.count("\n") / (len(actions) + 1))
                            if reward > model.max_fine_app:
                                reward = -1

                            real_rewards.append(reward)



                            predict_pattern = None
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
                                # print(predict_pattern)

                            rating = model.knn.predict(predict_pattern).item()
                            if ball_patterns(events):
                                rewards.append(-5)
                                print("ball pattern!")
                                # time.sleep(0.5)
                                break
                            ratings.append(rating)
                            if reward == 0:
                                rewards.append(-1.5)
                                break
                            if 1:
                                if len(np.unique(events)) == 1 and len(events) >= 3:
                                    rating = -0.1
                            if len(actions) > 1:
                                rating -= (model.knn_avg - 1)
                            else:
                                rating = 1
                            reward *= rating
                            rewards.append(reward)


                    if count >= model.match_max_size:
                        is_done = True

                _, Qval = model.forward(data, mask, T=temper)
                Qval = Qval.detach().cpu().numpy()[0]
                del data

                # if epoch < 2  and in_round_count < 25:
                if epoch == 0 and in_round_count < 30 :
                    send_rewards = rewards
                elif turn_flag == 0:
                    send_rewards = real_rewards
                else:
                    send_rewards = ratings
                if in_round_count % 200 == 0:
                    update_policy(model, send_rewards, log_probs, values, Qval, entropy_term, flag=True)
                else:
                    update_policy(model, send_rewards, log_probs, values, Qval, entropy_term)

                all_ratings.append(np.sum(ratings))
                all_rewards.append(np.sum(rewards))
                numsteps.append(len(actions))
                avg_numsteps.append(np.mean(numsteps))
                mean_rewards.append(np.mean(all_rewards))
                real.append(np.max(real_rewards))
                rating_plot.append(np.max(ratings))
                mean_real.append(np.mean(real_rewards))
                index_max = np.argmax(real_rewards)
                sys.stdout.write(
                    "Real reward : {}, Rating {},  comparisons : {}\n".format(
                        real_rewards[index_max],
                        ratings[index_max],
                        sum([t != "nop" for sub in comp_values for t in sub]),
                    )
                )
                if 1:
                    str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols, all_comps)
                    sys.stdout.write(f"Pattern: events = {events}, conditions = {str_pattern} index = {index}\n")
                    # sys.stdout.write(f"Pattern = {pattern} index = {index}\n")
                    # time.sleep(2)
                sys.stdout.write(
                    "episode: {}, index: {}, total reward: {}, average_reward: {}, length: {}\n".format(
                        in_round_count,
                        index,
                        np.round(np.max(rewards), decimals=3),
                        np.round(np.mean(all_rewards), decimals=3),
                        len(actions),
                    )
                )
                if model.count > 15:
                    print("\n\n\n---- Stopping early because of low log ----\n\n\n")
                    model.count = 0
                    # time.sleep(1)
                    for g in model.optimizer.param_groups:
                        g['lr'] *= 0.5
                    # input("Press Enter to continue...")
                    break


            rating_groups = [
                np.mean(rating_plot[t : t + GRAPH_VALUE])
                for t in range(0, len(rating_plot), GRAPH_VALUE)
            ]
            real_groups = [
                np.mean(real[t : t + GRAPH_VALUE])
                for t in range(0, len(real), GRAPH_VALUE)
            ]

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
            # ax1.set_xticks(locations, labels)
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
            ax2.plot()

            # if not os.path.exists("temper_graphs/temp_" + str(temp_given)):
            #     os.mkdir("temper_graphs/temp_" + str(temp_given) + "/")
            plt.savefig("Graphs/" + str(len(real)))
            plt.show()



            if False:
                after_epoch_test(best_pattern)
                with open("Data/Matches/allMatches.txt", "r") as f:
                    results.append(int(f.read().count("\n") / (max_len_best + 1)))
                os.remove("Data/Matches/allMatches.txt")
    if test_epcohs:
        print(results)
        plt.plot(results, "g")
        plt.show()


def predict_window(model, i, data):
    data_size = len(data)
    added_info_size = (model.match_max_size + 1) * (model.num_cols + 1)
    added_info_size_knn = (model.match_max_size + 1) * (6 + 1)
    old_desicions = torch.tensor([0] * added_info_size)
    data2 = torch.cat((data, torch.tensor([0] * added_info_size_knn)), dim=0)
    data = torch.cat((data, old_desicions.float()), dim=0)
    count = 0
    is_done = False
    events = []
    actions, rewards, action_types, all_conds, comp_values, ratings, all_comps = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    str_pattern = ""
    mask = torch.tensor([1.0] * (model.num_events + 1))
    pattern = None
    while not is_done:
        action, _, _, _ = model.get_event(data, i, mask.detach())
        data = data.clone()
        data[data_size + count * (model.num_cols + 1)] = model.embedding_desicions(
            torch.tensor(action)
        ).cuda()
        data2[data_size + count * (model.num_cols + 1)] = data[data_size + count * (model.num_cols + 1)]
        count += 1
        if action == model.num_events:
            # mask[-1] = mask[-1].clone() * 1.1
            if len(actions) != 0:
                ratings = ratings[:-1]
                data[data_size + count * (model.num_cols + 1)] = torch.tensor(0)
            break
        else:
            mask[action] = mask[action].clone() * 0.3

            event = new_mapping(action)
            events.append(event)
            mini_actions, _, comp_vals, conds, actions_vals, _, comps = model.get_cols_mini_actions(data.cuda())
            all_comps.append(comps)
            for j, action_val in enumerate(actions_vals):
                data = data.clone()
                try:
                    data[data_size + count * (model.num_cols + 1) + j + 1] = model.embedding_actions(
                        torch.tensor(action_val))
                    data2[data_size + count * (model.num_cols + 1) + j + 1] = data[data_size + count * (model.num_cols + 1) + j + 1]
                except Exception as e:
                    print(f"count {count}, j {j}")
            actions.append(mini_actions)

            currentPath = pathlib.Path(os.path.dirname(__file__))
            absolutePath = str(currentPath.parent)
            sys.path.append(absolutePath)
            file = os.path.join(absolutePath, "Model", "training", "{}.txt".format(i))

            # comp_vals = set_values(comp_vals, model.all_cols, mini_actions, event, conds, file)
            all_conds.append(conds)


            if comp_vals.count("nop") != len(comp_vals):
                # bayesian_dict = set_values_bayesian(comp_vals, model.all_cols, mini_actions, event, all_conds, file)
                bayesian_dict = set_values_bayesian(comp_vals,
                    model.all_cols, mini_actions, event,
                    all_conds, file, model.max_values_bayes,
                     model.min_values_bayes
                 )
                store_to_file(events, actions, i, comp_values, model.cols, all_conds, comp_vals, all_comps)
                b_optimizer = BayesianOptimization(
                    f=bayesian_function,
                    pbounds=bayesian_dict,
                    random_state=1,
                )
                try:
                    b_optimizer.maximize(
                        init_points=5,
                        n_iter=3,
                    )

                    selected_values = list(b_optimizer.max['params'].values())
                except Exception as e:
                    print(bayesian_dict)
                    selected_values = [max(model.normailze_values) for _ in range(len(bayesian_dict))]
                comp_vals = replace_values(comp_vals, selected_values)

            comp_values.append(comp_vals)
            # all_conds.append(conds)
            pattern = OpenCEP_pattern(
                events, actions, i, comp_values, model.cols, all_conds, all_comps
            )
            eff_pattern = pattern.condition
            with open("Data/Matches/{}Matches.txt".format(i), "r") as f:
                pattern_copy = (data[-added_info_size:]).detach().cpu().numpy().reshape(1,-1)
                rating = model.knn.predict(pattern_copy).item()

                reward = int(f.read().count("\n") / (len(actions) + 1))
                if 1:
                    reward *= rating
                    ratings.append(rating)
                    rewards.append(reward)
                    sys.stdout.write(f"Knn out: {rating}\n")
                str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols, all_comps)
            if count >= model.match_max_size:
                is_done = True

    if len(ratings) == 0:
        return [], [], -1, " "
    else:
        events_ball = [4 if event in [4,8,10] else event for event in events]
        if len(np.unique(events_ball)) == 1:
            ratings[-1] = 0
        return events, data[-added_info_size:], ratings[-1], str_pattern


def predict_patterns(model):
    def choose_best_pattern(patterns, ratings):
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        results = [0.0] * len(ratings)
        for i in range(0, len(ratings) - 1):
            for j in range(i, len(events)):
                sim = similar(patterns[i], patterns[j])
                results[i] += sim
                results[j] += sim
        results = torch.tensor(results) * torch.tensor(ratings)
        print(f"The similarities are:{results}")

        best_index = np.argmax(results)
        return best_index, results[best_index]

    model.eval()
    events = []
    patterns = []
    ratings = []
    pattern_strs = []
    # types = [] Todo: do this also
    for i, data in enumerate(model.data[-100:]):
        event, pattern, rating, pattern_str = predict_window(model, i, data)
        if len(event) != 0:
            # event = [str(val) for val in event]
            events.append(event)
            pattern = ",".join([str(i) for i in pattern])
            patterns.append(pattern)
            ratings.append(rating)
            pattern_strs.append(pattern_str)


    # values, counts = np.unique(events, return_counts=True)
    # print(counts)

    # print(values[np.argmax(counts)])
    print("Looking for most similar\n")
    best_index, best_sim = choose_best_pattern(patterns, ratings)
    print(best_sim)
    print(events[best_index])
    print(pattern_strs[best_index])

def main():
    class_inst = ruleMiningClass(data_path="Football/xaa", num_events=41,
                                 max_values=[97000, 100000, 15000, 20000, 20000, 20000],
                                 normailze_values=[24000, 45000, 6000, 9999, 9999, 9999])
    train(class_inst)
    predict_patterns(model=class_inst)


if __name__ == "__main__":
    torch.set_num_threads(30)
    main()
