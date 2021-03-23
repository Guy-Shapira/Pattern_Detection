import torch
import torch.nn as nn
import math
import os
from Patterns.utils import (
    to_var,
    mapping,
    OpenCEP_pattern,
    pattern_complexity,
    after_epoch_test,
    new_mapping,
    get_action_type,
    create_pattern_str,
    store_patterns_and_rating_to_csv,
)
import tqdm
import sys
import time
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import datetime
from difflib import SequenceMatcher
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
GRAPH_VALUE = 50
GAMMA = 0.99
EMBDEDDING_TOTAL_SIZE = 21
# EMBDEDDING_TOTAL_SIZE = 12


class genDataClass(nn.Module):
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
    ):
        super().__init__()
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
        self.hidden_size = 2048
        self.num_cols = num_cols
        self.num_actions = (3 * (self.match_max_size + 1)) * 2 * 2 + 1  # [>|<|= * [match_max_size + 1(value)] * not / reg] * (and/or) |nop
        # self.num_actions = self.value_options * self.num_events + 1
        self.embedding_actions = nn.Embedding(self.num_actions + 1, 1)
        self.embedding_desicions = nn.Embedding(self.num_events + 1, 1)
        self.actions = [">", "<", "="]
        # self.cols = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
        # self.cols  = ["x", "y", "z", "vx", "vy", "vz"]
        self.cols  = ["x", "y", "z", "vx", "vy"]
        # self.cols  = ["x", "y", "z"]



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
            if sliding_window_data is None:
                sliding_window_data = data[i : i + self.window_size]
                sliding_window_data = sliding_window_data.unsqueeze(0)
            else:
                to_add = data[i : i + self.window_size].unsqueeze(0)
                sliding_window_data = torch.cat((sliding_window_data, to_add))
        return sliding_window_data



def genData(model, num_epochs=10):
    torch.autograd.set_detect_anomaly(True)
    added_info_size = (model.match_max_size + 1) * (model.num_cols + 1) + 1 # need to remove + 1
    # added_info_size_knn = (model.match_max_size + 1) * (3 + 1) + 1
    results = []
    all_rewards = []
    numsteps = []
    avg_numsteps = []
    mean_rewards = []
    real, mean_real = [], []

    for epoch in range(num_epochs):
        pbar_file = sys.stdout
        with tqdm.tqdm(total=len(os.listdir("Model/training")[:500]), file=pbar_file) as pbar:
            for i, data in enumerate(model.data[epoch * 500 :(epoch + 1) * 500]):
                data_size = len(data)
                old_desicions = torch.tensor([0] * added_info_size)
                # data2 = torch.cat((data, torch.tensor([0] * added_info_size_knn).float()), dim=0)
                data = torch.cat((data, old_desicions.float()), dim=0)
                count = 0
                best_reward = 0.0
                pbar.update(n=1)
                is_done = False
                events = []
                actions, action_types, real_rewards, all_conds, comp_values, all_comps = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                values = []
                patterns = []
                ratings = []
                while not is_done:
                    if np.random.randint(10000 // (len(events) + 1)) < 5:
                        is_done = True
                    event = new_mapping(1, random=True)
                    action = new_mapping(event, reverse=True)
                    data[data_size + count * (model.num_cols + 1)] = model.embedding_desicions(
                        torch.tensor(action)
                    )
                    # data2[data_size + count * (model.num_cols + 1)] = data[data_size + count * (model.num_cols + 1)]
                    count += 1
                    if action == model.num_events:
                        if len(actions) == 0:
                            is_done = True
                            break
                    else:
                        events.append(event)
                        mini_actions = []
                        actions_vals = []
                        conds = []
                        comps = []
                        comp_vals = []
                        for col in range(len(model.cols)):
                            highest_prob_action = np.random.randint(model.num_actions)
                            # mini_action, cond = get_action_type(highest_prob_action, model.num_actions, model.actions)
                            mini_action, cond, comp_to = get_action_type(action, model.num_actions, model.actions, model.match_max_size)
                            comps.append(comp_to)
                            if np.random.randint(5) == 0:
                                mini_action ="nop"

                            highest_prob_value = "nop"
                            if len(mini_action.split("value")) > 1:
                                highest_prob_value = np.random.randint(model.max_values[col])

                            mini_action = mini_action.replace("value", "")  # TODO:replace this shit
                            mini_actions.append(mini_action)
                            actions_vals.append(highest_prob_action)
                            conds.append(cond)
                            comp_vals.append(highest_prob_value)


                        for j, action_val in enumerate(actions_vals):
                            data = data.clone()
                            try:
                                data[data_size + count * (model.num_cols + 1) + j + 1] = model.embedding_actions(
                                    torch.tensor(action_val))
                                # data2[data_size + count * (model.num_cols + 1) + j + 1] = data[data_size + count * (model.num_cols + 1) + j + 1]
                            except Exception as e:
                                print(f" len of arr {len(data)}")
                                print(f"index {data_size + count * (model.num_cols + 1) + j + 1}")
                                print(f"count {count}, j {j}")
                                print(e)
                                exit()
                        actions.append(mini_actions)
                        all_conds.append(conds)
                        all_comps.append(comps)
                        comp_values.append(comp_vals)

                        str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols, all_comps)
                        if len(events) > 1:
                            pass
                            sys.stdout.write(f"Pattern: events = {events}, conditions = {str_pattern}\n")

                        user_reward = 1
                        if len(events) == 1:
                            user_reward = np.random.uniform(0.8, 1.25)
                        else:
                            # print(events)
                            events_ball = [4 if event in [4,8,10] else event for event in events]
                            # print(events_ball)
                            unique, app_count = np.unique(events, return_counts=True)
                            for i in range(len(unique)):
                                if unique[i] != 4:
                                    app_count[i] += 1
                            for k in range(len(unique)):
                                user_reward += math.pow(0.5, k + 1) * app_count[k] * 1.5

                            user_reward += 0.25 * sum([t != "nop" for sub in comp_values for t in sub])

                            flatten = lambda list_list: [item for sublist in list_list for item in sublist]
                            flat_conds = flatten(all_conds)
                            if flat_conds.count("and") < (2/3) * len(flat_conds):
                                user_reward -= 1
                            # if flat_conds.count("or") > 5 :
                            #     user_reward -= 1.5
                            flat_actions = flatten(actions)
                            not_count = sum([str.startswith("not") for str in flat_actions])
                            unique, app_count = np.unique(flat_actions, return_counts=True)
                            for k in range(len(unique)):
                                if unique[k].startswith("not"):
                                    pass
                                    # user_reward -= math.pow(0.2, k+1) * app_count[k] * 1.5
                                else:
                                    user_reward += math.pow(0.2, k + 1) * app_count[k] * 1.25
                            user_reward -= 0.25 * not_count
                            ball_unique = np.unique(events_ball)
                            if len(events_ball) >= 3 and len(ball_unique) == 1:
                                user_reward = 0
                            else:
                                num_non_ball = sum([1 if event != 4 else 0 for event in events_ball])
                                if len(events_ball) >= 5 and num_non_ball <=2 :
                                    user_reward = 0

                        if np.random.randint(5) == 0:
                            store_patterns_and_rating_to_csv(data[-added_info_size:], user_reward , events, str_pattern)

                    if count >= model.match_max_size:
                        is_done = True


def main():
    class_inst = genDataClass(data_path="Football/xaa", num_events=41,
                                 max_values=[97000, 100000, 15000, 20000, 20000, 20000],
                                 normailze_values=[24000, 45000, 6000, 9999, 9999, 9999])
    genData(class_inst)


if __name__ == "__main__":
    torch.set_num_threads(10)
    main()
