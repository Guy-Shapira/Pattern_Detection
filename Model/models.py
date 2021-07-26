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

class Critic(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(Critic, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.critic_reward = nn.Linear(self.hidden_size2, 1).cuda()
        self.critic_rating = nn.Linear(self.hidden_size2, 1).cuda()


    def forward(self, actor_output):
        value_reward = self.critic_reward(actor_output)
        value_rating = self.critic_rating(actor_output)
        return value_reward, value_rating

class Actor(nn.Module):
    def __init__(self, hidden_size1, hidden_size2,
                embeddding_total_size, window_size, match_max_size,
                num_cols, num_actions, num_events):
        super(Actor, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.embeddding_total_size = embeddding_total_size
        self.window_size = window_size
        self.match_max_size = match_max_size
        self.num_cols = num_cols
        self.num_actions = num_actions
        self.num_events = num_events

        self.linear_base = nn.Linear(
            self.window_size * self.embeddding_total_size,
            self.hidden_size1,
        ).cuda()
        self.dropout = nn.Dropout(p=0.4).cuda()
        self.spread_patterns = nn.Linear(
            (self.match_max_size + 1) * (self.num_cols + 1),
            self.hidden_size1,
        ).cuda()
        self.linear_finish = nn.Linear(
            2 * self.hidden_size1,
            self.hidden_size2,
        ).cuda()
        self.event_tagger = nn.Linear(self.hidden_size2, self.num_events + 1).cuda()

        self.action_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size2, self.num_actions)
                for _ in range(self.num_cols)
            ]
        ).cuda()

    def forward(self, input, old_desicions, mask, training_factor, T):
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

        x1 = self.dropout(self.linear_base(input.cuda())).cuda()
        x2 = self.spread_patterns(old_desicions.cuda()).cuda()
        if np.random.rand() <= 1 - training_factor:
            x1 *= 0.1
            x2 *= 5.5
        combined = torch.cat((x1, x2)).cuda()
        after_relu = F.leaky_relu(self.linear_finish(combined))
        event_before_softmax = self.event_tagger(after_relu)
        if mask is None:
            event_after_softmax = event_before_softmax
        else:
            event_after_softmax = masked_softmax(event_before_softmax, mask.clone(), dim=0, T=T)
        return after_relu, event_after_softmax

    def forward_mini_actions(self, index, data, mask, training_factor, T=1):
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

        x = F.leaky_relu(self.action_layers[index](data))
        probs = masked_softmax(x, mask, dim=0)
        numpy_probs = probs.detach().cpu().numpy()

        entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2
        if np.random.rand() > 1 - training_factor:
            highest_prob_action = np.random.randint(len(probs))
        else:
            highest_prob_action = np.random.choice(
                self.num_actions, p=np.squeeze(numpy_probs)
            )
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action]).cpu()

        return highest_prob_action, log_prob, entropy



class ActorCriticModel(nn.Module):
    def __init__(
        self,
        num_events,
        match_max_size=8,
        window_size=350,
        num_cols=0,
        hidden_size1=512,
        hidden_size2=2048,
        embeddding_total_size=8
    ):
        super().__init__()
        self.embeddding_total_size = embeddding_total_size
        self.actions = [">", "<", "="]
        self.num_events = num_events
        self.match_max_size = match_max_size
        self.window_size = window_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_cols = num_cols
        self.num_actions = (len(self.actions) * (self.match_max_size + 1)) * 2 * 2 + 1  # [>|<|= * [match_max_size + 1(value)] * not / reg] * (and/or) |nop
        self.embedding_actions = nn.Embedding(self.num_actions + 1, 1)
        self.embedding_desicions = nn.Embedding(self.num_events + 1, 1)

        self.critic = self._create_critic()
        self.actor = self._create_actor()

    def _create_actor(self):
        return Actor(
            hidden_size1=self.hidden_size1,
            hidden_size2=self.hidden_size2,
            embeddding_total_size=self.embeddding_total_size,
            window_size=self.window_size,
            match_max_size=self.match_max_size,
            num_cols=self.num_cols,
            num_actions=self.num_actions,
            num_events=self.num_events
        )

    def _create_critic(self):
        return Critic(hidden_size1=self.hidden_size1, hidden_size2=self.hidden_size2)

    def forward_actor(self, input, old_desicions, mask=None, training_factor=0.0, T=1):
        return self.actor.forward(input, old_desicions, mask, training_factor, T)

    def forward_critic(self, actor_output):
        return self.critic.forward(actor_output)

    def forward_actor_mini_actions(self, index, data, mask, training_factor):
        return self.actor.forward_mini_actions(index, data, mask, training_factor)

    #
    #
    # def layers_based_hidden(self, hidden_size):
    #     self.hidden_size1 = hidden_size
    #     self.linear_base = nn.Linear(
    #         self.window_size * self.embeddding_total_size,
    #         self.hidden_size1,
    #     ).cuda()
    #     self.linear_finish = nn.Linear(
    #         self.hidden_size1 + (self.match_max_size + 1) * (self.num_cols + 1),
    #         self.hidden_size2,
    #     ).cuda()
    #
    # def layers_based_hidden2(self, hidden_size):
    #     self.linear_finish = nn.Linear(
    #         self.hidden_size1 + (self.match_max_size + 1) * (self.num_cols + 1),
    #         self.hidden_size2,
    #     ).cuda()
    #     self.event_tagger = nn.Linear(self.hidden_size2, self.num_events + 1).cuda()
    #
    #     self.action_layers = nn.ModuleList(
    #         [
    #             nn.Linear(self.hidden_size2, self.num_actions)
    #             for _ in range(self.num_cols)
    #         ]
    #     ).cuda()
    #     self.critic_reward = nn.Linear(self.hidden_size2, 1).cuda()
    #     self.critic_rating = nn.Linear(self.hidden_size2, 1).cuda()
    #
    # def layers_hidden(self, hidden_size1, hidden_size2):
    #     self.hidden_size1 = hidden_size1
    #     self.hidden_size2 = hidden_size2
    #     self.linear_base = nn.Linear(
    #         self.window_size * self.embeddding_total_size,
    #         self.hidden_size1,
    #     ).cuda()
    #     self.dropout = nn.Dropout(p=0.4).cuda()
    #     self.spread_patterns = nn.Linear(
    #         (self.match_max_size + 1) * (self.num_cols + 1),
    #         self.hidden_size1,
    #     ).cuda()
    #     self.linear_finish = nn.Linear(
    #         2 * self.hidden_size1,
    #         self.hidden_size2,
    #     ).cuda()
    #     self.event_tagger = nn.Linear(self.hidden_size2, self.num_events + 1).cuda()
    #
    #     self.action_layers = nn.ModuleList(
    #         [
    #             nn.Linear(self.hidden_size2, self.num_actions)
    #             for _ in range(self.num_cols)
    #         ]
    #     ).cuda()
    #     self.critic_reward = nn.Linear(self.hidden_size2, 1).cuda()
    #     self.critic_rating = nn.Linear(self.hidden_size2, 1).cuda()
