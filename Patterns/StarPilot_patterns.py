import torch
import torch.nn as nn
import math
import os
import random
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
    normalizeData,
)
import tqdm
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import datetime
from difflib import SequenceMatcher
import pandas as pd
GRAPH_VALUE = 50
GAMMA = 0.99
# EMBDEDDING_TOTAL_SIZE = 12
# SAME_EVENTS = {13 : [13, 14, 97, 98], 47: [47, 16], 49: [49, 88], 19: [19, 52],
#                 53: [53, 54], 23: [23, 24], 57: [57, 58], 59: [59, 28], 61: [61, 62, 99, 100],
#                 63: [63, 64], 65: [65, 66], 67: [67, 68], 69: [69, 38], 71: [71, 40],
#                 73: [73, 74], 75: [75, 44], 105: [105, 106], 4: [4], 8: [8], 10: [10], 12: [12]}

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
        # self.actions = [">", "<", "=", "+>", "->", "*="]
        self.actions = [">", "<", "="]
        self.all_actions = self.actions
        self.all_actions.extend(["not" + i for i in self.all_actions])
        self.all_actions.extend(["v" + i for i in self.all_actions])

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
        self.num_actions = (len(self.actions) * (self.match_max_size + 1)) * 2 * 2 + 1  # [>|<|= * [match_max_size + 1(value)] * not / reg] * (and/or) |nop
        self.embedding_actions = nn.Embedding(self.num_actions + 1, 1)
        self.embedding_desicions = nn.Embedding(self.num_events + 1, 1)
        self.cols  = ["x", "y", "z", "vx", "vy"]



    def _create_data(self, data_path):
        date_time_obj = None
        data = None
        with open(data_path) as f:
            for line in f:
                values = line.split("\n")[0]
                values = values.split(",")
                event = values[0]
                for k, v in zip(SAME_EVENTS.keys(), SAME_EVENTS.values()):
                    if int(event) in v:
                        event = str(k)
                        break
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



def genData(model, num_epochs=8):
    flatten = lambda list_list: [item for sublist in list_list for item in sublist]
    torch.autograd.set_detect_anomaly(True)
    added_info_size = (model.match_max_size + 1) * (model.num_cols + 1)  # need to remove + 1
    results = []
    numsteps = []
    avg_numsteps = []
    mean_rewards = []
    real, mean_real = [], []
    test1, test2 = [], []
    patterns_all = ["pass", "dribble", "two-run"] * 25 + ["random"] * 10

    # patterns_all = ["pass"] * 5 + ["random"] * 3 + ["dribble", "two-run"] * 3
    # patterns_all = ["pass", "run", "random", "two-run"]
    for epoch in range(num_epochs):
        pbar_file = sys.stdout
        with tqdm.tqdm(total=len(os.listdir("Model/training")[:800]), file=pbar_file) as pbar:
            for i, data in enumerate(model.data[epoch * 800 :(epoch + 1) * 800]):
                data_size = len(data)
                old_desicions = torch.tensor([0] * added_info_size)
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
                pattern_type = random.choice(patterns_all)
                comp_targets = [i for i in range(0, model.match_max_size)] + ["value"]

                if pattern_type == "pass":
                    passer_event = random.choice([13, 47, 49, 19, 53, 23, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 105])
                    ball_event = random.choice([4, 8, 10])
                    direction = random.choice(["x", "y", "z"])
                    events = [passer_event, ball_event, ball_event]
                    actions.append(["=", "=", "=", "nop", "nop"])
                    all_conds.append(["and", "and", "and", "or", "or"])
                    comp_values.append(["nop"] * 5)
                    all_comps.append([1, 1, 1, "" ,""])
                    next_targets = random.choices(comp_targets, k=len(model.cols))
                    next_actions = random.choices(model.all_actions, k=len(model.cols))
                    next_conds = random.choices(["and", "or"], k=len(model.cols))
                    next_comp_vals = ["value" if "v" in act else random.choice(range(model.match_max_size)) for act in actions[-1]]
                    next_targets[0] = 2
                    if direction == "x":
                        next_actions[0] = "<"
                    elif direction == "y":
                        next_actions[1] = "<"
                    else:
                        next_actions[2] = "<"

                    comp_values.append(next_comp_vals)
                    actions.append(next_actions)
                    all_comps.append(next_targets)
                    all_conds.append(next_conds)
                    next_targets = random.choices(comp_targets, k=len(model.cols))
                    next_actions = random.choices(model.all_actions, k=len(model.cols))
                    next_conds = random.choices(["and", "or"], k=len(model.cols))
                    actions.append(next_actions)
                    all_comps.append(next_targets)
                    all_conds.append(next_conds)
                    next_comp_vals = ["value" if "v" in act else random.choice(range(model.match_max_size)) for act in actions[-1]]
                    comp_values.append(next_comp_vals)

                    for i in range(len(events)):
                        str_pattern = create_pattern_str(events[:i+1], actions[:i+1], comp_values[:i+1], all_conds[:i+1], model.cols, all_comps[:i+1])
                        user_reward = np.random.uniform(3*(i+1), 6*(i+1))
                        if np.random.randint(4):
                            store_patterns_and_rating_to_csv(data[-added_info_size:], user_reward , events[:i+1], flatten(actions[:i+1]), flatten(all_conds[:i+1]), str_pattern)
                            test2.append(user_reward)
                    continue

                elif pattern_type == "dribble":
                    runner_event = random.choice([13, 47, 49, 19, 53, 23, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 105])
                    ball_event = random.choice([4, 8, 10])
                    events = [runner_event, ball_event, runner_event, ball_event]
                    # player first app
                    actions.append(["=", "=", "=", "nop", "nop"])
                    all_conds.append(["and", "and", "and", "or", "or"])
                    comp_values.append(["nop"] * 5)
                    all_comps.append([1, 1, 1, "", ""])

                    # ball first app
                    actions.append(random.choices(model.all_actions, k=len(model.cols)))
                    all_conds.append(random.choices(["and", "or"], k=len(model.cols)))
                    all_comps.append(random.choices(comp_targets, k=len(model.cols)))
                    next_comp_vals = ["value" if "v" in act else random.choice(range(model.match_max_size)) for act in actions[-1]]
                    comp_values.append(next_comp_vals)

                    # player second app
                    next_actions = random.choices(model.all_actions, k=len(model.cols))
                    next_actions[1] = ">"
                    next_comps = random.choices(comp_targets, k=len(model.cols))
                    next_comps[1] = 0
                    actions.append(next_actions)
                    all_comps.append(next_comps)
                    all_conds.append(random.choices(["and", "or"], k=len(model.cols)))
                    next_comp_vals = ["value" if "v" in act else random.choice(range(model.match_max_size)) for act in next_actions]
                    comp_values.append(next_comp_vals)

                    # ball second app
                    actions.append(["=", "=", "=", ">", "nop"])
                    all_conds.append(["and", "and", "and", "or", "or"])
                    comp_values.append(["nop"] * 5)
                    all_comps.append([2, 2, 2, 1, ""])
                    next_comp_vals = ["value" if "v" in act else random.choice(range(model.match_max_size)) for act in actions[-1]]
                    comp_values.append(next_comp_vals)


                    for i in range(len(events)):
                        str_pattern = create_pattern_str(events[:i+1], actions[:i+1], comp_values[:i+1], all_conds[:i+1], model.cols, all_comps[:i+1])
                        user_reward = np.random.uniform(5*(i+1), 5*(i+1))
                        if np.random.randint(4):
                            store_patterns_and_rating_to_csv(data[-added_info_size:], user_reward , events[:i+1], flatten(actions[:i+1]), flatten(all_conds[:i+1]), str_pattern)
                            test2.append(user_reward)
                    continue

                elif pattern_type == "two-run":
                    players = [13, 47, 49, 19, 53, 23, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 105]
                    first_player = random.choice(players)
                    players.remove(first_player)
                    second_player = random.choice(players)

                    # first player: first app
                    events = [first_player, second_player, first_player, second_player]
                    actions.append(random.choices(model.all_actions, k=len(model.cols)))
                    all_conds.append(random.choices(["and", "or"], k=len(model.cols)))
                    all_comps.append(random.choices(comp_targets, k=len(model.cols)))
                    next_comp_vals = ["value" if "v" in act else random.choice(range(model.match_max_size)) for act in actions[-1]]
                    comp_values.append(next_comp_vals)

                    # second player: first app
                    actions.append(random.choices(model.all_actions, k=len(model.cols)))
                    all_conds.append(random.choices(["and", "or"], k=len(model.cols)))
                    all_comps.append(random.choices(comp_targets, k=len(model.cols)))
                    next_comp_vals = ["value" if "v" in act else random.choice(range(model.match_max_size)) for act in actions[-1]]
                    comp_values.append(next_comp_vals)

                    # first player: second app
                    next_actions = random.choices(model.all_actions, k=len(model.cols))
                    next_actions[1] = ">"
                    next_actions[4] = "not<"
                    next_comps = random.choices(comp_targets, k=len(model.cols))
                    next_comps[1] = 0
                    next_comps[4] = 0
                    actions.append(next_actions)
                    all_comps.append(next_comps)
                    all_conds.append(random.choices(["and", "or"], k=len(model.cols)))
                    next_comp_vals = ["value" if "v" in act else random.choice(range(model.match_max_size)) for act in next_actions]
                    comp_values.append(next_comp_vals)

                    # second player: second app
                    next_actions = random.choices(model.all_actions, k=len(model.cols))
                    next_actions[1] = ">"
                    next_actions[4] = "not<"
                    next_comps = random.choices(comp_targets, k=len(model.cols))
                    next_comps[1] = 1
                    next_comps[4] = 1
                    actions.append(next_actions)
                    all_comps.append(next_comps)
                    all_conds.append(random.choices(["and", "or"], k=len(model.cols)))
                    next_comp_vals = ["value" if "v" in act else random.choice(range(model.match_max_size)) for act in actions[-1]]
                    comp_values.append(next_comp_vals)


                    for i in range(len(events)):
                        str_pattern = create_pattern_str(events[:i+1], actions[:i+1], comp_values[:i+1], all_conds[:i+1], model.cols, all_comps[:i+1])
                        user_reward = np.random.uniform(5*(i+1), 7*(i+1))
                        if np.random.randint(4):
                            store_patterns_and_rating_to_csv(data[-added_info_size:], user_reward , events[:i+1], flatten(actions[:i+1]), flatten(all_conds[:i+1]), str_pattern)
                            test2.append(user_reward)
                    continue

                elif pattern_type == "random":
                    while not is_done:
                        if np.random.randint(10000 // (len(events) + 1)) < 5:
                            is_done = True
                        event = new_mapping(1, random=True)
                        action = new_mapping(event, reverse=True)
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
                                highest_prob_action = np.random.randint(model.num_actions - 1)
                                mini_action, cond, comp_to = get_action_type(highest_prob_action, model.num_actions, model.actions, model.match_max_size)
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
                            actions.append(mini_actions)
                            all_conds.append(conds)
                            all_comps.append(comps)
                            comp_values.append(comp_vals)

                            str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols, all_comps)
                            if len(events) > 1:
                                pass

                            user_reward = 1
                            if len(events) == 1:
                                user_reward = np.random.uniform(0.8, 1.5)
                            else:
                                events_ball = [4 if event in [4,8,10] else event for event in events]
                                unique, app_count = np.unique(events, return_counts=True)
                                for i in range(len(unique)):
                                    if unique[i] != 4:
                                        app_count[i] += 1
                                for k in range(len(unique)):
                                    user_reward += math.pow(0.7, k + 1) * app_count[k] * 1.5

                                user_reward += 0.25 * sum([t != "nop" for sub in comp_values for t in sub])

                                flat_conds = flatten(all_conds)
                                if flat_conds.count("and") < 0.5 * len(flat_conds):
                                    user_reward -= -1.2 * (1 + flat_conds.count("or") / len(flat_conds))
                                flat_actions = flatten(actions)
                                not_count = sum([str.startswith("not") for str in flat_actions])
                                unique, app_count = np.unique(flat_actions, return_counts=True)

                                # user_reward -= 0.5 * not_count
                                ball_unique = np.unique(events_ball)
                                if len(events_ball) >= 3 and len(ball_unique) == 1:
                                    user_reward = np.random.uniform(0.3, 1.3)
                                else:
                                    num_non_ball = sum([1 if event != 4 else 0 for event in events_ball])
                                    if len(events_ball) >= 5 and num_non_ball <= 2 :
                                        user_reward = np.random.uniform(0.3, 1.3)

                        if (np.random.randint(5) or (len(test1) > 500 and user_reward > 10)):
                            if user_reward > 0:
                                store_patterns_and_rating_to_csv(data[-added_info_size:], user_reward , events, flatten(actions), flatten(all_conds), str_pattern)
                                test1.append(user_reward)
                        if count >= model.match_max_size:
                            is_done = True
    print(np.mean(test1))
    print(np.mean(test2))
    return test1



def main():
    class_inst = genDataClass(data_path="Football/xaa", num_events=20,
                                 max_values=[97000, 100000, 15000, 20000, 20000, 20000],
                                 normailze_values=[24000, 45000, 6000, 9999, 9999, 9999])
    test = genData(class_inst)
    normalizeData()
    print (np.mean(test))

if __name__ == "__main__":
    torch.set_num_threads(30)
    main()
