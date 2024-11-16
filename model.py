import torch
import torch.nn as nn
from env import Env, State

class Duel_DDNQ(nn.Module):
    def __init__(self, pretrained_model, LSTM_output_dim = 38, head_hidden_dim = 128):
        super(Duel_DDNQ, self).__init__()
        self.lstm = pretrained_model
        #tba, adjustment for the model's heads
        self.adv_fc = nn.Linear(LSTM_output_dim, head_hidden_dim)
        self.adv_out = nn.Linear(head_hidden_dim, Env.n_actions)
        self.val_fc = nn.Linear(LSTM_output_dim, head_hidden_dim)
        self.val_out = nn.Linear(head_hidden_dim, 1)


    def forward(self, x:State) ->float:
        player = (x.dealer+len(x.bidding_sequence))%4
        cnt = player
        single_bid = torch.zeros(183, dtype=torch.float32)
        single_bid[:66] = x.features[player]
        tensor_list  = []
        while cnt-x.dealer <= len(x.bidding_sequence):
            _row = single_bid.clone
            for i in range(3, 0, -1):
                if cnt - i < 0:
                    _row[66+(3-i)*39] = 1
                else:
                    _row[67+(3-i)*39+x.bidding_sequence[cnt-i]] = 1
            cnt+=4
            tensor_list.append(_row)
        shared_out = self.lstm(tensor_list.cuda())
        adv = nn.ReLU(self.adv_fc(shared_out))
        adv = self.adv_out(adv)
        val = nn.ReLU(self.val_fc(shared_out))
        val = self.val_out(val)
        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        #tba, adjustment for Q-value calc
        return q_values.cpu()