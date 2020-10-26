import torch
import torch.nn as nn
import os
from Model.utils import prepare_loss_clac, prepare_pattern, to_var, mapping, OpenCEP_pattern
import tqdm
import sys
import time
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import datetime
date_time_str = '2020-10-26 13:40:00.243860'
date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')

GAMMA = 0.99
with torch.autograd.set_detect_anomaly(True):

    class ruleMiningClass(nn.Module):
        def __init__(self, data_path, num_events, match_max_size=8, max_values=3000, window_size=20, max_count=3000):
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
            self.value_options = 4
            self.num_actions = self.value_options * self.num_events + 1
            # first self.num_events are just select events
            # last is nop action version
            # then every follow self.num events is select or select with value compare to previous action (for now only direct previous)
            self.linear1_action = nn.Linear(self.window_size * 9, self.hidden_size)
            self.linear2_action = nn.Linear(self.hidden_size, self.num_actions)
            self.critic = nn.Linear(self.hidden_size, 1)
            #TODO: add follow option, maybe double the num action, so it would be action + follow/not follow
            # needs to be smarter if follow is not possible
            self._create_training_dir(data_path)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05)

        def _create_data(self, data_path):
            data = None
            with open(data_path) as f:
                for line in f:
                    event, value, count = line.split(",")
                    event = ord(event) - ord("A")
                    event = self.embedding_events(torch.tensor(event))
                    value = self.embedding_values(torch.tensor(int(value)))
                    count = count[:-1]
                    count = datetime.datetime.strptime(count, '%Y-%m-%d %H:%M:%S.%f')
                    count -= date_time_obj
                    count = count.total_seconds()
                    count = self.embedding_count(torch.tensor(int(count)))
                    if data is None:
                        data = torch.cat((event, value, count), 0)
                        data = data.unsqueeze(0)
                    else:
                        new_data = torch.cat((event, value, count), 0)
                        new_data = new_data.unsqueeze(0)
                        data = torch.cat((data, new_data), 0)

            sliding_window_data = None
            for i in range(0, len(data) - self.window_size):
                if sliding_window_data is None:
                    sliding_window_data = data[i: i + self.window_size]
                    sliding_window_data = sliding_window_data.unsqueeze(0)
                else:
                    to_add = data[i: i + self.window_size].unsqueeze(0)
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

        def forward(self, input, T=5):
            x = F.relu(self.linear1_action(input))
            value = self.critic(x)
            x = self.linear2_action(x)
            x = F.softmax(x / T, dim=0)

            return x, value

        def get_action(self, input, T=1):
            probs, value = self.forward(Variable(input), T=T)
            numpy_probs = probs.detach().numpy()
            highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(numpy_probs))
            log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
            entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs))
            return highest_prob_action, log_prob, value, entropy


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

        policy_network.zero_grad()
        policy_gradient.backward()
        policy_network.optimizer.step()

    def train(model, num_epochs=1):
        all_rewards = []
        numsteps = []
        avg_numsteps = []
        temper = 1
        mean_rewards = []
        for epoch in range(num_epochs):
            pbar_file = sys.stdout
            with tqdm.tqdm(total=len(os.listdir("Model/training")[:50]), file=pbar_file) as pbar:
                for i, data in enumerate(model.data[:50]):
                    if i % 10 == 0:
                        temper *= 1.25
                    count = 0
                    best_reward = 0.0
                    pbar.update(n=1)
                    is_done = False
                    actions = []
                    rewards = []
                    log_probs = []
                    action_types = []
                    values = []
                    entropy_term = 0
                    real_rewards = []
                    while not is_done:
                        data = model.data[i + count]
                        action, log_prob, value, entropy = model.get_action(data, T=temper)
                        count += 1
                        value = value.detach().numpy()[0]
                        entropy_term += entropy
                        if action == model.num_actions - 1:
                            values.append(value)
                            if len(actions) == 0:
                                log_probs.append(log_prob)
                                rewards.append(-1.1)
                                # break
                            else:
                                log_probs.append(log_prob)
                                rewards.append(rewards[-1])
                                break

                        else:
                            action, kind_of_action = mapping(model.num_events, action)
                            actions.append(action)
                            values.append(value)
                            action_types.append(kind_of_action)
                            log_probs.append(log_prob)
                            OpenCEP_pattern(actions, action_types, i)
                            with open("Data/Matches/{}Matches.txt".format(i), "r") as f:
                                reward = int(f.read().count("\n"))
                                real_rewards.append(reward)
                                if reward == 0:
                                    is_done = True
                                    rewards.append(-1.5)
                                    break
                                reward *= 1.25
                                reward += len(actions) * 0.1
                                reward += len(np.where(np.array(action_types) != 'nop')[0]) * 1.5
                                rewards.append(reward)
                            if reward > best_reward:
                                best_reward = reward
                                copyfile("pattern", "best_pattern/best_pattern{}".format(i))
                            os.remove("Data/Matches/{}Matches.txt".format(i))
                        if count >= model.match_max_size:
                            is_done = True

                    Qval, _ = model.forward(data, T=temper)
                    Qval = Qval.detach().numpy()[0]
                    update_policy(model, rewards, log_probs, values, Qval, entropy_term)
                    all_rewards.append(np.sum(rewards))
                    numsteps.append(len(actions))
                    avg_numsteps.append(np.mean(numsteps))
                    mean_rewards.append(np.mean(all_rewards))
                    sys.stdout.write("Real reward : {}\n".format(np.max(real_rewards)))
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(i, np.round(np.sum(rewards), decimals=3),  np.round(np.mean(all_rewards), decimals=3), len(actions)))
                plt.plot(all_rewards)
                plt.plot(mean_rewards, 'g')
                plt.xlabel('Episode')
                plt.show()


    class_inst = ruleMiningClass(data_path="Data/train_data_stream.txt", num_events=6)
    train(class_inst)