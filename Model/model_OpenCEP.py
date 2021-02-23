import torch
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
PAD_VALUE = -5.5
# EMBDEDDING_TOTAL_SIZE = 12


class ruleMiningClass(nn.Module):
    # TODO: switch all max_inputs to lists of len num_cols
    def __init__(
        self,
        data_path,
        num_events,
        match_max_size=10,
        max_values=None,
        normailze_values=None,
        window_size=150,
        max_count=2000,
        num_cols=2,
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
        self.num_actions = (3 * 3 * 2) + 1 # [>|<|= * 3(reg, neg, value)| (and then or)] | nop
        # self.num_actions = self.value_options * self.num_events + 1
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
          # This is probably very shite
        # TODO: add follow option, maybe double the num action, so it would be action + follow/not follow
        # needs to be smarter if follow is not possible

        self.value_layer = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.max_values[i]) for i in range(self.num_cols)]
        )
        # self._create_training_dir(data_path)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        self.actions = [">", "<", "="]
        # self.cols = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
        # self.cols  = ["x", "y", "z", "vx", "vy", "vz"]
        self.cols  = ["x", "y", "z"]

        self.knn = self._create_df()


    def _create_df(self):
        df = pd.read_csv("Patterns/pattern3.csv")[["pattern", "rating"]]
        df.pattern = df.pattern.apply(lambda x : x.split(","))
        df.pattern = df.pattern.apply(lambda x : [float(i) for i in x])
        df.rating = df.rating.apply(lambda x : round(float(x) + 0.5))
        df_patterns = pd.DataFrame(df.pattern.values.tolist())
        df_patterns.replace(0, PAD_VALUE, inplace=True)
        df_patterns = df_patterns.iloc[:, :-1]
        print(df_patterns[:10])
        knn = KNN(n_neighbors=1)
        knn.fit(df_patterns, df["rating"])
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

    def get_event(self, input, mask=None, T=1):
        probs, value = self.forward(Variable(input), mask, T=T)
        numpy_probs = probs.detach().cpu().numpy()
        action = np.random.choice(
            self.num_events + 1, p=np.squeeze(numpy_probs)
        )
        log_prob = torch.log(probs.squeeze(0)[action])
        entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2
        return action, log_prob, value, entropy

    def get_value(self, input):
        x = F.relu(self.linear_base(Variable(input)))
        x = self.value_layer(x)
        probs = F.softmax(x, dim=0)
        numpy_probs = probs.detach().numpy()
        highest_prob_action = np.random.choice(
            self.max_values, p=np.squeeze(numpy_probs)
        )
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def single_col_mini_action(self, data, index):
        # replace get event
        x = F.relu(self.action_layers[index](data))
        probs = F.softmax(x, dim=0)
        numpy_probs = probs.detach().cpu().numpy()

        highest_prob_action = np.random.choice(
            self.num_actions, p=np.squeeze(numpy_probs)
        )
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action]).cpu()
        highest_prob_value = None
        mini_action, _ = get_action_type(
            highest_prob_action, self.num_actions, self.actions
        )
        entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2

        if len(mini_action.split("value")) > 1:
            value_layer = self.value_layer[index].cpu()
            value_probs = F.softmax(value_layer(data.cpu()), dim=0)
            numpy_probs = value_probs.detach().cpu().numpy()
            highest_prob_value = np.random.choice(
                self.max_values[index], p=np.squeeze(numpy_probs)
            )
            highest_prob_value -= self.normailze_values[index]
            log_prob += torch.log(value_probs.squeeze(0)[highest_prob_action]).cpu()
        return highest_prob_action, highest_prob_value, log_prob, entropy

    def get_cols_mini_actions(self, data):
        mini_actions = []
        log_probs = 0.0
        compl_vals = []
        conds = []
        mini_actions_vals = []
        total_entropy = 0
        updated_data = self.linear_base(data)
        for i in range(self.num_cols):
            # TODO: save return values and stuff
            action, value, log, entropy = self.single_col_mini_action(updated_data, i) #this is weird, should update data after actions
            mini_actions_vals.append(action)
            total_entropy += entropy / self.num_cols
            mini_action, cond = get_action_type(action, self.num_actions, self.actions)
            conds.append(cond)
            if len(mini_action.split("value")) > 1:
                mini_action = mini_action.replace("value", "")  # TODO:replace this shit
                compl_vals.append(value)  # TODO: change
            else:
                compl_vals.append("nop")  # TODO: change
            mini_actions.append(mini_action)
            log_probs += log / self.num_cols
        return mini_actions, log_probs, compl_vals, conds, mini_actions_vals, total_entropy


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


def update_policy(policy_network, rewards, log_probs, values, Qval, entropy_term):
    Qvals = np.zeros_like(values)
    for t in reversed(range(len(rewards))):
        Qval = rewards[t] + GAMMA * Qval
        Qvals[t] = Qval

    values = torch.FloatTensor(values)
    Qvals = torch.FloatTensor(Qvals)
    log_probs = torch.stack(log_probs)

    advantage = Qvals - values
    actor_loss = (-log_probs * advantage.mean())
    critic_loss = 0.5 * advantage.pow(2).mean()
    print(f"Critic: {critic_loss}, Actor {actor_loss}\n")
    policy_gradient = actor_loss + critic_loss + 0.001 * entropy_term
    policy_gradient = policy_gradient.cuda()
    print(policy_gradient)
    policy_network.zero_grad()
    policy_gradient.mean().backward(retain_graph=True)
    policy_network.optimizer.step()


def train(model, num_epochs=5, test_epcohs=False):
    torch.autograd.set_detect_anomaly(True)
    # model = model.cuda()
    added_info_size = (model.match_max_size + 1) * (model.num_cols + 1)
    added_info_size_knn = (model.match_max_size + 1) * (3 + 1)
    oniline_model = nn.Linear(
                added_info_size, 1
    )


    online_loss = nn.MSELoss(reduction='mean')
    user_inputs , online_patterns  = [] , []
    total_user_inputs , total_online_patterns  = [] , []
    online_optimizer = torch.optim.SGD(oniline_model.parameters(), lr=0.001)
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
    switch_to_online = False
    for epoch in range(num_epochs):
        if epoch < 10:
            temper = 100
        else:
            temper = 5

        pbar_file = sys.stdout
        with tqdm.tqdm(total=len(os.listdir("Model/training")[:500]), file=pbar_file) as pbar:
            for i, data in enumerate(model.data[epoch * 500 :(epoch + 1) * 500]):
                data_size = len(data)
                old_desicions = torch.tensor([PAD_VALUE] * added_info_size)
                data2 = torch.cat((data, torch.tensor([PAD_VALUE] * added_info_size_knn).float()), dim=0)
                data = torch.cat((data, old_desicions.float()), dim=0)
                count = 0
                best_reward = 0.0
                pbar.update(n=1)
                is_done = False
                events = []
                actions, rewards, log_probs, action_types, real_rewards, all_conds = (
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
                while not is_done:
                    data = data.cuda()
                    mask_orig = mask.clone()
                    action, log_prob, value, entropy = model.get_event(
                        data, mask, T=temper
                    )
                    data = data.clone()
                    data[data_size + count * (model.num_cols + 1)] = model.embedding_desicions(
                        torch.tensor(action)
                    ).cuda()
                    data2[data_size + count * (model.num_cols + 1)] = data[data_size + count * (model.num_cols + 1)]
                    if i % 50 == 0 and epoch < 10:
                        temper /= 1.35
                    count += 1
                    value = value.detach().cpu().numpy()[0]
                    values.append(value)
                    entropy_term += entropy
                    if action == model.num_events:
                        mask[-1] = mask[-1].clone() * 1.1
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
                        mask[action] = mask[action].clone() * 0.3

                        event = new_mapping(action)
                        events.append(event)
                        mini_actions, log, comp_vals, conds, actions_vals, entropy = model.get_cols_mini_actions(data)
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
                        comp_values.append(comp_vals)
                        all_conds.append(conds)
                        pattern = OpenCEP_pattern(
                            events, actions, i, comp_values, model.cols, all_conds
                        )
                        str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols)
                        if len(events) > 1:
                            sys.stdout.write(f"Pattern: events = {events}, conditions = {str_pattern}\n")

                        eff_pattern = pattern.condition
                        with open("Data/Matches/{}Matches.txt".format(i), "r") as f:
                            reward = int(f.read().count("\n") / (len(actions) + 1))
                            real_rewards.append(reward)
                            pattern_copy = (data[-added_info_size:]).detach().cpu().numpy().reshape(1,-1)
                            rating = model.knn.predict(pattern_copy).item()
                            # if len(events) >= 3:
                            #     rating *= np.power(1.05, len(events) - 2)
                            #     if len(np.unique(events)) > 2 and len(events) >= 4:
                            #         rating *= 2
                            #     elif len(np.unique(events)) == 1:
                            #         rating *= 0.1
                            # reward = rating * 10
                            ratings.append(rating)
                            if reward == 0:
                                rewards.append(-1.5)
                                break
                            if 1:
                                reward *= rating
                                rewards.append(reward)
                                sys.stdout.write(f"Knn out: {rating}\n")


                                if 0 :
                                    if switch_to_online:
                                        user_reward = oniline_model(torch.tensor(data[-added_info_size:]).reshape(1,-1)).item()
                                        sys.stdout.write(f"predicted: {user_reward}")
                                    else:
                                        sys.stdout.write("Insert complexity rank:")
                                        user_reward = ""
                                        while user_reward == "":
                                            user_reward = input()
                                        user_reward = float(user_reward)
                                        online_patterns.append(data[-added_info_size:])
                                        user_inputs.append(user_reward)
                                        store_patterns_and_rating_to_csv(data[-added_info_size:]
                                        , user_reward, events, str_pattern)
                                        reward *= user_reward

                        if reward > best_reward:
                            best_reward = reward
                            copyfile(
                                "Data/Matches/{}Matches.txt".format(i),
                                "best_pattern/best_pattern{}".format(i),
                            )
                            total_best = reward
                        os.remove("Data/Matches/{}Matches.txt".format(i))
                        if reward > total_best:
                            with open("best.txt", "a+") as f:
                                bindings = [
                                    event + " as " + chr(ord("a") + k)
                                    for k, event in enumerate(actions)
                                ]
                                f.write("----\n")
                                f.write(",".join(bindings) + "\n")
                                f.write(str(eff_pattern) + "\n")
                                f.write("----")
                    if count >= model.match_max_size:
                        is_done = True

                _, Qval = model.forward(data, mask, T=temper)
                Qval = Qval.detach().cpu().numpy()[0]
                del data


                update_policy(model, rewards, log_probs, values, Qval, entropy_term)
                all_ratings.append(np.sum(ratings))
                all_rewards.append(np.sum(rewards))
                numsteps.append(len(actions))
                avg_numsteps.append(np.mean(numsteps))
                mean_rewards.append(np.mean(all_rewards))
                real.append(np.max(real_rewards))
                rating_plot.append(np.max(ratings))
                mean_real.append(np.mean(real_rewards))
                sys.stdout.write(
                    "Real reward : {}, comparisons : {}\n".format(
                        np.max(real_rewards),
                        sum([i != "nop" for sub in comp_values for i in sub]),
                    )
                )
                if i % 2 == 0:
                    str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols)
                    sys.stdout.write(f"Pattern: events = {events}, conditions = {str_pattern}\n")
                sys.stdout.write(
                    "episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(
                        i,
                        np.round(np.sum(rewards), decimals=3),
                        np.round(np.mean(all_rewards), decimals=3),
                        len(actions),
                    )
                )
                # if len(user_inputs) >= 8 and not switch_to_online:
                #     online_patterns = torch.stack(online_patterns).reshape(len(user_inputs), -1)
                #     user_inputs = torch.FloatTensor(user_inputs).reshape(len(user_inputs), -1)
                #
                #     if total_user_inputs == []:
                #         total_user_inputs = user_inputs
                #     else:
                #         total_user_inputs = torch.cat((total_user_inputs, user_inputs), dim=0)
                #
                #     if total_online_patterns == []:
                #         total_online_patterns = online_patterns
                #     else:
                #         total_online_patterns = torch.cat((total_online_patterns, online_patterns), dim=0)
                #
                #     predictions = oniline_model(total_online_patterns)
                #     loss_val = online_loss(predictions, total_user_inputs)
                #     if loss_val <= 0.5:
                #         switch_to_online = True
                #     sys.stdout.write(f"Online learning: loss = {loss_val.item()} \n")
                #     for (pred, true_val) in zip(predictions, total_user_inputs):
                #         sys.stdout.write(f"pred = {pred.item()} | input = {true_val.item()}\n")
                #     online_optimizer.zero_grad()
                #     loss_val.backward(retain_graph=True)
                #     online_optimizer.step()
                #     online_patterns = []
                #     user_inputs = []
            # rating_groups = [
            #     np.mean(rating_plot[i : i + GRAPH_VALUE])
            #     for i in range(0, len(rating_plot), GRAPH_VALUE)
            # ]
            # real_groups = [
            #     np.mean(real[i : i + GRAPH_VALUE])
            #     for i in range(0, len(real), GRAPH_VALUE)
            # ]
            #
            # fig, (ax1, ax2) = plt.subplots(2)
            # print("\n----3")
            #
            # ax1.set_xlabel("Episode")
            # ax1.set_title("Reward vs number of episodes played")
            # labels = [
            #     "{}-{}".format(i, i + GRAPH_VALUE)
            #     for i in range(0, len(real), GRAPH_VALUE)
            # ]
            # locations = [
            #     i + int(GRAPH_VALUE / 2) for i in range(0, len(real), GRAPH_VALUE)
            # ]
            # plt.sca(ax1)
            # plt.xticks(locations, labels)
            #
            # ax1.scatter(locations, real_groups, c="g")
            # # ax1.set_xticks(locations, labels)
            # ax1.set_ylabel("Avg Matches per window")
            #
            # ax1.plot()
            #
            # locations = [
            #     i + int(GRAPH_VALUE / 2) for i in range(0, len(rating_plot), GRAPH_VALUE)
            # ]
            # # ax2.xticks(locations, labels)
            # ax2.set_ylabel("Avg Rating per window")
            # ax2.set_xlabel("Episode")
            # ax2.set_title("Rating vs number of episodes played")
            # plt.sca(ax2)
            # plt.xticks(locations, labels)
            #
            # ax2.scatter(locations, rating_groups, c="g")
            # # ax2.set_xticks(locations, labels)
            # ax2.plot()
            #
            # # plt.savefig('myfig')
            # # plt.show()



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
    added_info_size_knn = (model.match_max_size + 1) * (3 + 1)
    old_desicions = torch.tensor([0] * added_info_size)
    data2 = torch.cat((data, torch.tensor([0] * added_info_size_knn)), dim=0)
    data = torch.cat((data, old_desicions.float()), dim=0)
    count = 0
    is_done = False
    events = []
    actions, rewards, action_types, all_conds, comp_values, ratings = (
        [],
        [],
        [],
        [],
        [],
        []
    )
    str_pattern = ""
    mask = torch.tensor([1.0] * (model.num_events + 1))
    pattern = None
    while not is_done:
        action, _, _, _ = model.get_event(data, mask.detach())
        data = data.clone()
        data[data_size + count * (model.num_cols + 1)] = model.embedding_desicions(
            torch.tensor(action)
        ).cuda()
        data2[data_size + count * (model.num_cols + 1)] = data[data_size + count * (model.num_cols + 1)]
        count += 1
        if action == model.num_events:
            mask[-1] = mask[-1].clone() * 1.1
            if len(actions) != 0:
                ratings = ratings[:-1]
                data[data_size + count * (model.num_cols + 1)] = torch.tensor(0)
            break
        else:
            mask[action] = mask[action].clone() * 0.3

            event = new_mapping(action)
            events.append(event)
            mini_actions, _, comp_vals, conds, actions_vals, _ = model.get_cols_mini_actions(data.cuda())
            for j, action_val in enumerate(actions_vals):
                data = data.clone()
                try:
                    data[data_size + count * (model.num_cols + 1) + j + 1] = model.embedding_actions(
                        torch.tensor(action_val))
                    data2[data_size + count * (model.num_cols + 1) + j + 1] = data[data_size + count * (model.num_cols + 1) + j + 1]
                except Exception as e:
                    print(f"count {count}, j {j}")
            actions.append(mini_actions)
            comp_values.append(comp_vals)
            all_conds.append(conds)
            pattern = OpenCEP_pattern(
                events, actions, i, comp_values, model.cols, all_conds
            )
            eff_pattern = pattern.condition
            with open("Data/Matches/{}Matches.txt".format(i), "r") as f:
                reward = int(f.read().count("\n") / (len(actions) + 1))
                if reward == 0:
                    ratings.append(0)
                    break


                str_pattern = create_pattern_str(events, actions, comp_values, all_conds, model.cols)
                if len(events) > 1:
                    if 1:
                        pattern_copy = (data2[-added_info_size_knn:]).detach().cpu().numpy().reshape(1,-1)
                        rating = model.knn.predict(pattern_copy).item()
                        ratings.append(rating)
                else:
                    ratings.append(1)
            if count >= model.match_max_size:
                is_done = True

    if len(ratings) == 0:
        return [], [], -1, " "
    else:
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
    for i, data in enumerate(model.data[:100]):
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
    class_inst = ruleMiningClass(data_path="Football/merge_x00", num_events=41,
                                 max_values=[97000, 100000, 15000, 20000, 20000, 20000],
                                 normailze_values=[24000, 45000, 6000, 9999, 9999, 9999])
    train(class_inst)
    # predict_patterns(model=class_inst)


if __name__ == "__main__":
    torch.set_num_threads(20)
    main()
