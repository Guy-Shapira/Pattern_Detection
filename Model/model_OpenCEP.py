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


GRAPH_VALUE = 10
GAMMA = 0.99

EMBDEDDING_TOTAL_SIZE = 15


class ruleMiningClass(nn.Module):
    # TODO: switch all max_inputs to lists of len num_cols
    def __init__(
        self,
        data_path,
        num_events,
        match_max_size=10,
        max_values=2000,
        window_size=30,
        max_count=2000,
        num_cols=3,
    ):
        super().__init__()
        self.num_events = num_events
        self.match_max_size = match_max_size
        self.max_values = max_values
        self.window_size = window_size
        self.embedding_events = nn.Embedding(num_events + 1, 3)
        self.embedding_values = nn.Embedding(max_values, 3)
        self.embedding_count = nn.Embedding(max_count, 3)
        self.data = self._create_data(data_path)
        self.data = self.data.view(len(self.data), -1)
        self.hidden_size = 2048
        self.num_cols = num_cols
        self.num_actions = 3 * 3 + 1 * 2  # [>|<|= * 3(reg, neg, value)| nop] (and  then or)
        # self.num_actions = self.value_options * self.num_events + 1
        self.embedding_desicions = nn.Embedding(self.num_actions, 1)
        self.linear_base = nn.Linear(
            self.window_size * EMBDEDDING_TOTAL_SIZE + self.match_max_size,
            self.hidden_size,
        )
        self.event_tagger = nn.Linear(self.hidden_size, self.num_events + 1)

        self.action_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.num_actions * 2)
                for _ in range(self.num_cols)
            ]
        )
        self.critic = nn.Linear(self.hidden_size, 1)  # This is probably very shite
        # TODO: add follow option, maybe double the num action, so it would be action + follow/not follow
        # needs to be smarter if follow is not possible

        self.value_layer = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.max_values) for _ in range(self.num_cols)]
        )
        self._create_training_dir(data_path)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        self.actions = [">", "<", "="]
        self.cols = ["Value1", "Value2", "Value3"]

    def _create_data(self, data_path):
        date_time_obj = None
        data = None
        with open(data_path) as f:
            for line in f:
                event, value1, value2, value3, count = line.split(",")
                event = ord(event) - ord("A")
                event = self.embedding_events(torch.tensor(event))
                value1 = self.embedding_values(torch.tensor(int(value1)))
                value2 = self.embedding_values(torch.tensor(int(value2)))
                value3 = self.embedding_values(torch.tensor(int(value3)))
                count = count[:-1]
                count = datetime.datetime.strptime(count, "%Y-%m-%d %H:%M:%S.%f")
                if date_time_obj == None:
                    date_time_obj = count
                count -= date_time_obj
                count = count.total_seconds()
                count = self.embedding_count(torch.tensor(int(count)))
                if data is None:
                    data = torch.cat((event, value1, value2, value3, count), 0)
                    data = data.unsqueeze(0)
                else:
                    new_data = torch.cat((event, value1, value2, value3, count), 0)
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
            masked_vec = vec * mask.float()
            max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
            exps = torch.exp(masked_vec - max_vec)
            masked_exps = exps * mask.float()
            masked_sums = masked_exps.sum(dim, keepdim=True)
            zeros = masked_sums == 0
            masked_sums += zeros.float()
            return masked_exps / masked_sums

        x = F.relu(self.linear_base(input))

        value = self.critic(x)
        x = self.event_tagger(x)
        x = masked_softmax(x, mask.clone(), dim=0, T=T)
        # return x
        return x, value

    def get_event(self, input, mask=None, T=1):
        probs, value = self.forward(Variable(input), mask, T=T)
        numpy_probs = probs.detach().numpy()
        action = np.random.choice(
            self.num_events + 1, p=np.squeeze(numpy_probs)
        )
        log_prob = torch.log(probs.squeeze(0)[action])
        entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7))
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
        numpy_probs = probs.detach().numpy()

        highest_prob_action = np.random.choice(
            self.num_actions * 2, p=np.squeeze(numpy_probs)
        )
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        highest_prob_value = None
        mini_action, _ = get_action_type(
            highest_prob_action, self.num_actions, self.actions
        )
        if len(mini_action.split("value")) > 1:
            value_probs = F.softmax(self.value_layer[index](data), dim=0)
            numpy_probs = value_probs.detach().numpy()
            highest_prob_value = np.random.choice(
                self.max_values, p=np.squeeze(numpy_probs)
            )
            log_prob += torch.log(value_probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, highest_prob_value, log_prob

    def get_cols_mini_actions(self, data):
        mini_actions = []
        log_probs = 0.0
        compl_vals = []
        conds = []
        updated_data = self.linear_base(data)
        for i in range(self.num_cols):
            # TODO: save return values and stuff
            action, value, log = self.single_col_mini_action(updated_data, i)
            mini_action, cond = get_action_type(action, self.num_actions, self.actions)
            conds.append(cond)
            if len(mini_action.split("value")) > 1:
                mini_action = mini_action.replace("value", "")  # TODO:replace this shit
                compl_vals.append(value)  # TODO: change
            else:
                compl_vals.append("nop")  # TODO: change
            mini_actions.append(mini_action)
            log_probs += log / self.num_cols
        return mini_actions, log_probs, compl_vals, conds


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
    actor_loss = (-log_probs * advantage).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()
    policy_gradient = actor_loss + critic_loss + 0.001 * entropy_term
    print(policy_gradient)
    policy_network.zero_grad()
    policy_gradient.backward(retain_graph=True)
    policy_network.optimizer.step()


def train(model, num_epochs=5, test_epcohs=False):
    results = []
    total_best = -1
    all_rewards = []
    max_len_best = -1
    numsteps = []
    avg_numsteps = []
    temper = 1
    mean_rewards = []
    real, mean_real = [], []
    best_pattern = None
    entropy_term = 0
    event_training = 1
    for epoch in range(num_epochs):
        pbar_file = sys.stdout
        with tqdm.tqdm(
            total=len(os.listdir("Model/training")[:100]), file=pbar_file
        ) as pbar:
            for i, data in enumerate(model.data[epoch * 100 :(epoch + 1) * 100]):
                if i % 5 == 0:
                    event_training = 1 -event_training
                data_size = len(data)
                old_desicions = torch.tensor([0] * model.match_max_size)
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
                temper = 1
                mask = torch.tensor([1.0] * (model.num_events + 1))
                while not is_done:
                    action, log_prob, value, entropy = model.get_event(
                        data, mask.detach(), T=temper
                    )
                    data = data.clone()
                    data[data_size + count] = model.embedding_desicions(
                        torch.tensor(action)
                    )
                    # if i % 50 == 0:
                    #     temper *= 1.05
                    count += 1
                    value = value.detach().numpy()[0]
                    values.append(value)
                    entropy_term += entropy
                    if action == model.num_events:
                        mask[-1] = mask[-1].clone() * 1.1
                        if len(actions) == 0:
                            log_probs.append(log_prob)
                            rewards.append(-1.5)
                            real_rewards.append(-1.5)
                        else:
                            log_probs.append(log_prob)
                            rewards.append(rewards[-1])
                            break
                    else:
                        mask[-1] = mask[-1].clone() * 1.3
                        mask[action] = mask[action].clone() * 0.8

                        event = new_mapping(action)
                        events.append(event)
                        mini_actions, log, comp_vals, conds = model.get_cols_mini_actions(data)
                        # if event_training:
                        #     log_prob = log
                        log_prob += log
                        log_probs.append(log_prob)
                        actions.append(mini_actions)
                        comp_values.append(comp_vals)
                        all_conds.append(conds)
                        pattern = OpenCEP_pattern(
                            events, actions, i, comp_values, model.cols, all_conds
                        )
                        eff_pattern = pattern.condition
                        with open("Data/Matches/{}Matches.txt".format(i), "r") as f:
                            reward = int(f.read().count("\n") / (len(actions) + 1))
                            real_rewards.append(reward)
                            if reward == 0:
                                rewards.append(-1.5)
                                break
                            reward *= pattern_complexity(events, actions, comp_values, model.num_events, model.num_actions)
                            # TODO: need to design new fitness function
                            rewards.append(reward)
                        if reward > best_reward:
                            best_reward = reward
                            copyfile(
                                "Data/Matches/{}Matches.txt".format(i),
                                "best_pattern/best_pattern{}".format(i),
                            )
                            max_len_best = len(actions)
                            total_best = reward
                            best_pattern = pattern
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
                Qval = Qval.detach().numpy()[0]
                update_policy(model, rewards, log_probs, values, Qval, entropy_term)
                update_policy1(model, rewards, log_probs)
                all_rewards.append(np.sum(rewards))
                numsteps.append(len(actions))
                avg_numsteps.append(np.mean(numsteps))
                mean_rewards.append(np.mean(all_rewards))
                real.append(np.max(real_rewards))
                mean_real.append(np.mean(real_rewards))
                sys.stdout.write(
                    "Real reward : {}, comparisons : {}\n".format(
                        np.max(real_rewards),
                        sum([i != "nop" for sub in comp_values for i in sub]),
                    )
                )
                if i % 10 == 0:
                    sys.stdout.write(f"Pattern: events = {events}, conditions = {eff_pattern}\n")
                sys.stdout.write(
                    "episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(
                        i,
                        np.round(np.sum(rewards), decimals=3),
                        np.round(np.mean(all_rewards), decimals=3),
                        len(actions),
                    )
                )

            real_groups = [
                np.mean(real[i : i + GRAPH_VALUE])
                for i in range(0, len(real), GRAPH_VALUE)
            ]
            plt.xlabel("Episode")
            labels = [
                "{}-{}".format(i, i + GRAPH_VALUE)
                for i in range(0, len(real), GRAPH_VALUE)
            ]
            locations = [
                i + int(GRAPH_VALUE / 2) for i in range(0, len(real), GRAPH_VALUE)
            ]
            plt.scatter(locations, real_groups, c="g")
            plt.xticks(locations, labels)
            plt.ylabel("Matches per window")
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
    old_desicions = torch.tensor([0] * model.match_max_size)
    data = torch.cat((data, old_desicions.float()), dim=0)
    count = 0
    is_done = False
    actions, action_types, comp_values = [], [], []
    mask = torch.tensor([1.0] * model.num_events + 1)
    pattern = None
    while not is_done:
        action, _, _, _ = model.get_event(data, mask.detach())
        data = data.clone()
        data[data_size + count] = model.embedding_desicions(torch.tensor(action))
        count += 1
        if action == model.num_actions - 1:
            is_done = True
        else:
            index_mod = action % model.num_events
            index_mod = torch.tensor(
                (
                    [1.0] * (index_mod)
                    + [0.95]
                    + [1.0] * (model.num_events - index_mod - 1)
                )
                * model.value_options
                + [1.5]
            )
            mask *= index_mod
            mask[-1] = mask[-1].clone() * 1.5
            action, kind_of_action = mapping(model.num_events, action)
            if len(kind_of_action.split("value")) > 1:
                comp_value, _ = model.get_value(data)
                kind_of_action = kind_of_action.replace("value", "")
                comp_values.append(comp_value)
            else:
                comp_values.append("none")
            actions.append(action)
            action_types.append(kind_of_action)
            pattern = OpenCEP_pattern(actions, action_types, i, comp_values)
            os.remove("Data/Matches/{}Matches.txt".format(i))
        if count >= model.match_max_size:
            is_done = True
    return actions, pattern


def predict_patterns(model):
    def choose_best_pattern(events):
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        results = [0.0] * len(events)
        for i in range(0, len(events) - 1):
            for j in range(i, len(events)):
                sim = similar(events[i], events[j])
                results[i] += sim
                results[j] += sim
        return np.argmax(results)

    model.eval()
    events = []
    patterns = []
    # types = [] Todo: do this also
    for i, data in enumerate(model.data[:100]):
        event, pattern = predict_window(model, i, data)
        events.append("".join(event))
        patterns.append(pattern)

    values, counts = np.unique(events, return_counts=True)
    print(counts)

    print(values[np.argmax(counts)])

    best_index = choose_best_pattern(events)
    print(events[best_index])
    print(events)
    avg_len = sum([len(e) for e in events]) / len(events)
    print(avg_len)
    after_epoch_test(patterns[best_index])
    with open("Data/Matches/allMatches.txt", "r") as f:
        print(int(f.read().count("\n") / (len(events[best_index]) + 1)))


def main():
    class_inst = ruleMiningClass(data_path="Data/train_data_stream.txt", num_events=6)
    train(class_inst)
    # predict_patterns(model=class_inst)


if __name__ == "__main__":
    main()
