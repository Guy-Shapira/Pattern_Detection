import torch
import torch.nn as nn
import os
from utils import prepare_loss_clac, prepare_pattern, to_var, fix_res
import tqdm
import sys
import time
import torch.nn.functional as F

class ruleMiningClass(nn.Module):
    def __init__(self, data_path, num_events, match_max_size=5, max_values=1000, window_size=20, max_count=1000):
        super().__init__()
        self.num_events = num_events
        self.match_max_size = match_max_size
        self.max_values= max_values
        self.window_size= window_size
        self.embedding_events = nn.Embedding(num_events + 1, 3)
        self.embedding_values = nn.Embedding(max_values, 3)
        self.embedding_count = nn.Embedding(max_count, 3)
        # For now it's shit just to have something to work on quickly as possiable
        self.data = self._create_data(data_path)
        self.data = self.data.view(len(self.data), -1)
        weight_size = 32
        emb_size = 9
        self.emb_size = emb_size
        hidden_size = 128
        self.eff_window_size = self.window_size + 1
        self.answer_seq_len = self.match_max_size
        self.is_GRU = False
        self.hidden_size = hidden_size
        self.enc = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.dec = nn.LSTMCell(emb_size, hidden_size) # LSTMCell's input is always batch first

        self.W1 = nn.Linear(hidden_size, weight_size, bias=False) # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False) # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False) # scaling sum of enc and dec by v.T
        # self.network = self._create_network()
        self._create_training_dir(data_path)

    def _create_data(self, data_path):
        data = None
        event = self.embedding_events(torch.tensor(self.num_events))
        value = self.embedding_values(torch.tensor(0))
        count = self.embedding_count(torch.tensor(0))
        fake_data = torch.cat((event, value, count), 0)
        fake_data = fake_data.unsqueeze(0)
        with open(data_path) as f:
            for line in f:
                event, value, count = line.split(",")
                if event == "A":
                    event = 0
                elif event == "B":
                    event = 1
                elif event == "C":
                    event = 2
                elif event == "D":
                    event = 3
                elif event == "E":
                    event = 4
                elif event == "F":
                    event = 5
                event = self.embedding_events(torch.tensor(event))
                value = self.embedding_values(torch.tensor(int(value)))
                count = self.embedding_count(torch.tensor(int(count)))
                if data is None:
                    data = torch.cat((event, value, count), 0)
                    data = data.unsqueeze(0)
                    # print(data)
                    # exit()

                else:
                    new_data = torch.cat((event, value, count), 0)
                    new_data = new_data.unsqueeze(0)
                    data = torch.cat((data, new_data), 0)

        sliding_window_data = None
        for i in range(0, len(data) - self.window_size):
            if sliding_window_data is None:
                sliding_window_data = data[i: i + self.window_size]
                sliding_window_data = torch.cat((sliding_window_data, fake_data), dim=0)
                sliding_window_data = sliding_window_data.unsqueeze(0)
            else:
                to_add = torch.cat((data[i: i + self.window_size], fake_data), dim=0).unsqueeze(0)
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
                f.write("G,0,0")

    # def _create_network(self):
    #     encoder_states, hc = self.enc(input) # encoder_state: (bs, L, H)
    #     # return nn.Sequential(
    #     #     nn.Linear(
    #     #         in_features=self.window_size * 9,
    #     #         out_features=self.window_size * 3,
    #     #     ),
    #     #     nn.LeakyReLU(),
    #     #     nn.Dropout(),
    #     #     nn.Linear(
    #     #         in_features=self.window_size * 3,
    #     #         out_features=self.match_max_size * (self.num_events + 1),
    #     #     ),
    #     #     nn.LeakyReLU(),
    #     # )

    def forward(self, input):
        softMax = nn.Softmax(dim=0)
        batch_size = 1
        input = input.view(1, self.eff_window_size, 9)
        encoder_states, hc = self.enc(input) # encoder_state: (bs, L, H)
        encoder_states = encoder_states.transpose(1, 0) # (L, bs, H)

        # Decoding states initialization
        decoder_input = to_var(torch.zeros(batch_size, self.emb_size)) # (bs, embd_size)
        hidden = to_var(torch.zeros([batch_size, self.hidden_size]))   # (bs, h)
        cell_state = encoder_states[-1]                                # (bs, h)

        probs = []
        # Decoding
        for i in range(self.answer_seq_len): # range(M)
            if self.is_GRU:
                hidden = self.dec(decoder_input, hidden) # (bs, h), (bs, h)
            else:
                hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (bs, h), (bs, h)

            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_states)          # (L, bs, W)
            blend2 = self.W2(hidden)                  # (bs, W)
            blend_sum = F.tanh(blend1 + blend2)    # (L, bs, W)
            out = self.vt(blend_sum).squeeze()        # (L, bs)
            # out = F.log_softmax(out.contiguous(), -1) # (bs, L)
            out = softMax(out)
            out = torch.multinomial(out, num_samples=1)
            probs.append(out)

        probs = torch.stack(probs, dim=1)           # (bs, M, L)

        return probs
        # softMax = nn.Softmax(dim=1)
        # for_res = self.network(x)
        # for_res = for_res.view(-1, self.num_events + 1)
        # # print(for_res.shape)
        # res = torch.multinomial(softMax(for_res), num_samples=1)
        # return res


def train(model, num_epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(num_epochs):
        total_loss = 0.0
        loss = torch.tensor(0.0, requires_grad=True)
        optimizer.zero_grad()
        pbar_file = sys.stdout
        with tqdm.tqdm(total=len(os.listdir("Model/training")), file=pbar_file) as pbar:
            for i, data in enumerate(model.data):
                pbar.update(n=1)
                res = model.forward(data)
                res = fix_res(res, "Model/training/{}.txt".format(i))

                try:
                    prepare_pattern(res, model.num_events, i)
                    match_size = prepare_loss_clac(res, i, model.num_events, model.window_size)
                    if match_size == 0:
                        match_size = 0.5
                    cmd = '"C:/Users/User/.jdks/jdk-11/bin/java.exe" -jar out/artifacts/rule_mining_jar/rule_mining.jar >/nul 2>&1'
                    os.system(cmd)
                    with open("Data/{}.txt".format(i), "r") as f:
                        value = int(f.read()) + 1
                        value += 1 / match_size
                        loss = loss + torch.tensor(1 / value, requires_grad=True)

                except:
                    exit()
                if i % 25 == 0 and i != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.65)
                    print(loss)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss = torch.tensor(0.0, requires_grad=True)
            print(total_loss)
                # TODO - also define and calc the loss function and of course backwords and shit






class_inst = ruleMiningClass(data_path="Data/train_data_stream.txt", num_events=6)
train(class_inst)