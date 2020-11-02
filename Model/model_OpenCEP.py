import torch
import torch.nn as nn
import os
from Model.utils import prepare_loss_clac, prepare_pattern, to_var, mapping, OpenCEP_pattern, pattern_complexity
import tqdm
import sys
import time
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import datetime

GAMMA = 0.90
with torch.autograd.set_detect_anomaly(True):

    class ruleMiningClass(nn.Module):
        def __init__(self, data_path, num_events, match_max_size=9, max_values=5000, window_size=30, max_count=5000):
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
            self.value_options = 4 * 2 * 2 #not support and value support
            self.num_actions = self.value_options * self.num_events + 1
            self.embedding_desicions = nn.Embedding(self.num_actions, 1)
            self.linear1_action = nn.Linear((self.window_size + 1) * 9, self.hidden_size)
            self.linear2_action = nn.Linear(self.hidden_size, self.num_actions)
            self.critic = nn.Linear(self.hidden_size, 1)
            #TODO: add follow option, maybe double the num action, so it would be action + follow/not follow
            # needs to be smarter if follow is not possible

            self.value_layer = nn.Linear(self.hidden_size, self.max_values)
            self._create_training_dir(data_path)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)

        def _create_data(self, data_path):
            date_time_obj = None
            data = None
            with open(data_path) as f:
                for line in f:
                    event, value, count = line.split(",")
                    event = ord(event) - ord("A")
                    event = self.embedding_events(torch.tensor(event))
                    value = self.embedding_values(torch.tensor(int(value)))
                    count = count[:-1]
                    count = datetime.datetime.strptime(count, '%Y-%m-%d %H:%M:%S.%f')
                    if date_time_obj == None:
                        date_time_obj = count
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

        # def forward(self, input, T=5):
        #     x = F.relu(self.linear1_action(input))
        #     value = self.critic(x)
        #     x = self.linear2_action(x)
        #     x = F.softmax(x / T, dim=0)
        #
        #     return x, value
        def forward(self, input, mask, T=5):
            def masked_softmax(vec, mask, dim=1, T=5):
                vec = vec / T
                masked_vec = vec * mask.float()
                max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
                exps = torch.exp(masked_vec-max_vec)
                masked_exps = exps * mask.float()
                masked_sums = masked_exps.sum(dim, keepdim=True)
                zeros = (masked_sums == 0)
                masked_sums += zeros.float()
                return masked_exps/masked_sums

            x = F.relu(self.linear1_action(input))

            value = self.critic(x)
            x = self.linear2_action(x)
            x = masked_softmax(x, mask.clone(), dim=0, T=T)

            return x, value

        def get_action(self, input, mask=None, T=1):
            probs, value = self.forward(Variable(input), mask, T=T)
            numpy_probs = probs.detach().numpy()
            highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(numpy_probs))
            log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
            entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs))
            return highest_prob_action, log_prob, value, entropy

        def get_value(self, input):
            x = F.relu(self.linear1_action(Variable(input)))
            x = self.value_layer(x)
            probs = F.softmax(x, dim=0)
            numpy_probs = probs.detach().numpy()
            highest_prob_action = np.random.choice(self.max_values, p=np.squeeze(numpy_probs))
            log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
            return highest_prob_action, log_prob


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

    def train(model, num_epochs=5):
        all_rewards = []
        numsteps = []
        avg_numsteps = []
        temper = 1
        mean_rewards = []
        real, mean_real = [], []
        for epoch in range(num_epochs):
            pbar_file = sys.stdout
            with tqdm.tqdm(total=len(os.listdir("Model/training")[:250]), file=pbar_file) as pbar:
                for i, data in enumerate(model.data[:250]):
                    data_size = len(data)
                    old_desicions = torch.tensor([0] * model.match_max_size)
                    data = torch.cat((data, old_desicions.float()), dim=0)
                    if i % 50 == 0:
                        temper *= 1.05
                    count = 0
                    best_reward = 0.0
                    pbar.update(n=1)
                    is_done = False
                    actions, rewards, log_probs, action_types, values, real_rewards = [], [], [] ,[], [], []
                    comp_values = []
                    entropy_term = 0
                    mask = torch.tensor([1.0] * model.num_actions)
                    while not is_done:
                        action, log_prob, value, entropy = model.get_action(data, mask.detach(), T=temper)
                        data = data.clone()
                        data[data_size + count] = model.embedding_desicions(torch.tensor(action))
                        count += 1
                        value = value.detach().numpy()[0]
                        entropy_term += entropy
                        if action == model.num_actions - 1:
                            # mask[-1] = mask[-1].clone() * 1.1
                            values.append(value)
                            if len(actions) == 0:
                                log_probs.append(log_prob)
                                rewards.append(-1.1)
                                real_rewards.append(-1.1)
                            else:
                                log_probs.append(log_prob)
                                rewards.append(rewards[-1])
                                break
                        else:
                            mask[action] = mask[action].clone() * 0.8
                            # mask[-1] = mask[-1].clone() * 1.3
                            action, kind_of_action = mapping(model.num_events, action)
                            if len(kind_of_action.split("value")) > 1:
                                comp_value, log_prob_2 = model.get_value(data)
                                kind_of_action = kind_of_action.replace("value", "")
                                log_prob += log_prob_2 #This might be really bad
                                comp_values.append(comp_value)
                            else:
                                comp_values.append("none")
                            actions.append(action)
                            values.append(value)
                            action_types.append(kind_of_action)
                            log_probs.append(log_prob)
                            OpenCEP_pattern(actions, action_types, i, comp_values)
                            with open("Data/Matches/{}Matches.txt".format(i), "r") as f:
                                reward = int(f.read().count("\n") / (len(actions) + 1))
                                real_rewards.append(reward)
                                if reward == 0:
                                    is_done = True
                                    rewards.append(-1.5)
                                    break
                                reward *= pattern_complexity(actions, action_types, comp_values, model.num_events, model.value_options)
                                rewards.append(reward)
                            if reward > best_reward:
                                best_reward = reward
                                copyfile("Data/Matches/{}Matches.txt".format(i), "best_pattern/best_pattern{}".format(i))
                            os.remove("Data/Matches/{}Matches.txt".format(i))
                        if count >= model.match_max_size:
                            is_done = True

                    Qval, _ = model.forward(data, mask, T=temper)
                    Qval = Qval.detach().numpy()[0]
                    update_policy(model, rewards, log_probs, values, Qval, entropy_term)
                    all_rewards.append(np.sum(rewards))
                    numsteps.append(len(actions))
                    avg_numsteps.append(np.mean(numsteps))
                    mean_rewards.append(np.mean(all_rewards))
                    real.append(np.sum(real_rewards))
                    mean_real.append(np.mean(real_rewards))
                    sys.stdout.write("Real reward : {}, compersions : {}\n".format(np.max(real_rewards), len(np.where(np.array(comp_values) != 'none')[0])))
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(i, np.round(np.sum(rewards), decimals=3),  np.round(np.mean(all_rewards), decimals=3), len(actions)))
                # plt.plot(mean_real)
                plt.plot(mean_rewards, 'g')
                plt.xlabel('Episode')
                plt.show()


    class_inst = ruleMiningClass(data_path="Data/train_data_stream.txt", num_events=5)
    train(class_inst)