import torch
import torch.nn as nn
from env import Env

class Duel_DDNQ(nn.Module):
    def __init__(self, pretrained_model, LSTM_output_dim = 38, head_hidden_dim = 128):
        super(Duel_DQNQ, self).__init__()
        self.lstm = pretrained_model
        
        self.adv_fc = nn.Linear(LSTM_output_dim, head_hidden_dim)
        self.adv_out = nn.Linear(head_hidden_dim, Env.action_space)
        
        self.val_fc = nn.Linear(LSTM_output_dim, head_hidden_dim)
        self.val_out = nn.Linear(head_hidden_dim, 1)


    def forward(self, x):
        shared_out = self.lstm(x)
        
        adv = nn.ReLU(self.adv_fc(shared_out))
        adv = self.adv_out(adv)
        
        val = nn.ReLU(self.val_fc(shared_out))
        val = self.val_out(val)
        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        return q_values