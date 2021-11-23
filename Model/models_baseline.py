import torch
import random
import torch.nn as nn
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from math import log, sqrt
import torch.nn.functional as F
from Model.utils import (
    OverFlowError,
)



torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class ModelBase(nn.Module):
    def __init__(self, hidden_size1, hidden_size2,
                embeddding_total_size, window_size, match_max_size,
                num_cols, num_actions, num_events):
        super(ModelBase, self).__init__()
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
        self.linear_finish = nn.Linear(
            self.hidden_size1,
            self.hidden_size2,
        ).cuda()

    def forward_base(self, input, training_factor):
        x1 = self.dropout(self.linear_base(input.cuda())).cuda()
        after_relu = F.leaky_relu(self.linear_finish(x1))
        return after_relu


class Critic(nn.Module):
    def __init__(self, hidden_size2):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size2

        self.critic_reward = nn.Linear(self.hidden_size, 1).cuda()
        self.critic_rating = nn.Linear(self.hidden_size, 1).cuda()


    def forward(self, base_output):
        value_reward = self.critic_reward(base_output)
        value_rating = self.critic_rating(base_output)
        return value_reward, value_rating

class Actor(ModelBase):
    def __init__(self, hidden_size1, hidden_size2,
                embeddding_total_size, window_size, match_max_size,
                num_cols, num_actions, num_events):
        super().__init__(hidden_size1, hidden_size2,
                    embeddding_total_size, window_size, match_max_size,
                    num_cols, num_actions, num_events)

        # (num_events + 1) * (Sum_{i=0}^{i=num_cols} {(num_actions)^i})
        # self.action_layer_size = (num_events + 1) * sum([num_actions ** i for i in range(0, num_cols)])
        self.action_layer = nn.Linear(self.hidden_size2, num_actions).cuda()

    def forward(self, input, softmax_flag, training_factor):

        base_output = self.forward_base(input, training_factor)
        action = self.action_layer(base_output)

        if not softmax_flag:
            return action, None
        else:
            m = nn.Softmax(dim=0).to(action.device)
            action = m(action)
            return base_output, action

class ActorCriticModel(nn.Module):
    def __init__(
        self,
        num_events,
        match_max_size=8,
        window_size=350,
        num_cols=0,
        hidden_size1=512,
        hidden_size2=2048,
        embeddding_total_size=8,
        num_actions=0
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
        # self.num_actions = (len(self.actions) * (self.match_max_size + 1)) * 2 * 2 + 1  # [>|<|= * [match_max_size + 1(value)] * not / reg] * (and/or) |nop
        self.num_actions = num_actions
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
        return Critic(
            hidden_size2=self.hidden_size2,
        )

    def forward_actor(self, input, softmax_flag=True, training_factor=0.0):
        return self.actor.forward(input, softmax_flag, training_factor)

    def forward_critic(self, base_output):
        return self.critic.forward(base_output)
