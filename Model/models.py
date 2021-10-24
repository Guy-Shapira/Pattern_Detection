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
        self.spread_patterns = nn.Linear(
            (self.match_max_size + 1) * (self.num_cols + 1),
            self.hidden_size1,
        ).cuda()
        self.linear_finish = nn.Linear(
            2 * self.hidden_size1,
            self.hidden_size2,
        ).cuda()

    def forward_base(self, input, old_desicions, training_factor):
        x1 = self.dropout(self.linear_base(input.cuda())).cuda()
        x2 = self.spread_patterns(old_desicions.cuda()).cuda()
        if np.random.rand() > 1 - training_factor:
            x1 *= 0.1
            x2 *= 5.5
        combined = torch.cat((x1, x2)).cuda()
        after_relu = F.leaky_relu(self.linear_finish(combined))
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


        self.event_tagger = nn.Linear(self.hidden_size2, self.num_events + 1).cuda()

        self.action_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size2, self.num_actions)
                for _ in range(self.num_cols)
            ]
        ).cuda()

    def forward(self, input, old_desicions, softmax_flag, training_factor):
        def masked_softmax(vec, mask, dim=1, T=1, epsilon=1e-5):
            vec = vec / T
            normalized_vec = vec - torch.max(vec)
            exps = torch.exp(normalized_vec).cpu()
            masked_exps = exps * mask.float()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            check_sums = masked_sums.cpu().detach().numpy()
            check_exps = masked_exps.cpu().detach().numpy()
            try:
                check_toghter = check_exps/check_sums
            except Exception as e:
                print(e)
                print("anomaly- extreamly large value!")
                #TODO: raise a unique error- to end training! - Done
                raise OverFlowError(vec, mask, T)
                # base_array = torch.zeros_like(masked_exps).to(masked_exps.device)
                # max_index = torch.argmax(masked_exps).item()
                # base_array[max_index] = 1.0
                # return base_array
                
            return (masked_exps/masked_sums)


        base_output = self.forward_base(input, old_desicions, training_factor)
        event_before_softmax = self.event_tagger(base_output)

        if not softmax_flag:
            return base_output, None
            # event_after_softmax = event_before_softmax
        else:
            # event_after_softmax = masked_softmax(event_before_softmax, mask.clone(), dim=0, T=T)
            m = nn.Softmax(dim=0).to(event_before_softmax.device)
            event_after_softmax = m(event_before_softmax)
            return base_output, event_after_softmax

    def forward_mini_actions(self, index, data, training_factor, action_counter, count_comparisons):
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
        m = nn.Softmax(dim=0).to(x.device)
        # probs = masked_softmax(x, mask, dim=0)
        probs = m(x)
        numpy_probs = probs.detach().cpu().numpy()

        entropy = -np.sum(np.mean(numpy_probs) * np.log(numpy_probs + 1e-7)) / 2

        numpy_probs = np.squeeze(numpy_probs).astype(float)
        if index > 1:

            ucb_factor = np.array([sqrt((2 * log(count_comparisons))/ (action_counter[i])) for i, _ in enumerate(numpy_probs)])

            # ucb_factor = np.array([sqrt((2 * log(action_counter))/ (count_comparisons[i])) for i, _ in enumerate(numpy_probs)])
            ucb_factor = ucb_factor / np.sum(ucb_factor)
            # numpy_probs = np.array([prob + sqrt((2 * log(self.count_events))/ (self.event_counter[i])) for i, prob in enumerate(numpy_probs)])
            numpy_probs += ucb_factor

            numpy_probs = numpy_probs / np.sum(numpy_probs)

        highest_prob_action = np.random.multinomial(
                n=1, pvals=numpy_probs, size=1
        )
        highest_prob_action = np.argmax(highest_prob_action)

        # if np.random.rand() > 1 - training_factor:
        #     highest_prob_action = np.random.randint(len(probs))
        # else:
        #     highest_prob_action = np.random.choice(
        #         self.num_actions, p=np.squeeze(numpy_probs)
        #     )
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
        return Critic(
            hidden_size2=self.hidden_size2,
        )

    def forward_actor(self, input, old_desicions, softmax_flag=True, training_factor=0.0):
        return self.actor.forward(input, old_desicions, softmax_flag, training_factor)

    def forward_critic(self, base_output):
        return self.critic.forward(base_output)

    def forward_actor_mini_actions(self, index, data, training_factor, action_counter, count_comparisons):
        return self.actor.forward_mini_actions(index, data, training_factor, action_counter, count_comparisons)
