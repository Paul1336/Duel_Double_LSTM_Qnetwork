import torch
import torch.nn as nn
from env import Env
from state import State
import logger

class Duel_DDNQ(nn.Module):
    """def __init__(self, pretrained_model, LSTM_output_dim = 38, head_hidden_dim = 128):
        super(Duel_DDNQ, self).__init__()
        self.lstm = nn.LSTM(input_size=183, hidden_size=LSTM_output_dim, batch_first=True)
        self.lstm.load_state_dict(pretrained_model)
        self.lstm = self.lstm.cuda()
        #pretrained_model.cuda()
        #tba, adjustment for the model's heads
        self.adv_fc = nn.Linear(LSTM_output_dim, head_hidden_dim).cuda()
        self.adv_out = nn.Linear(head_hidden_dim, Env.n_actions).cuda()
        self.val_fc = nn.Linear(LSTM_output_dim, head_hidden_dim).cuda()
        self.val_out = nn.Linear(head_hidden_dim, 1).cuda()
    """
    def __init__(self, pretrained_model_state_dict, lstm_hidden=1024, lstm_layers=4, LSTM_output_dim=38, head_hidden_dim=256):
        super(Duel_DDNQ, self).__init__()
        self.log = logger.get_logger(__name__)
        self.lstm = nn.LSTM(
            input_size=183,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_state_dict = {
            key.replace("lstm1.", ""): value
            for key, value in pretrained_model_state_dict.items()
            if key.startswith("lstm1.")
        }

        self.lstm.load_state_dict(lstm_state_dict)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, LSTM_output_dim),
            nn.LayerNorm(LSTM_output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc[0].weight.data = pretrained_model_state_dict["fc.weight"].clone()
        self.fc[0].bias.data = pretrained_model_state_dict["fc.bias"].clone()

        self.adv = nn.Sequential(
            nn.Linear(LSTM_output_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, Env.n_actions)
        )

        self.val = nn.Sequential(
            nn.Linear(LSTM_output_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1)
        )
        self.to(self.device)
        #self.fc = nn.Linear(1024, LSTM_output_dim)
        #self.fc.weight.data = pretrained_model_state_dict["fc.weight"].clone()
        #self.fc.bias.data = pretrained_model_state_dict["fc.bias"].clone()
        #self.fc = self.fc.cuda()
        
        #self.adv_fc = nn.Linear(LSTM_output_dim, head_hidden_dim).cuda()
        #self.adv_out = nn.Linear(head_hidden_dim, Env.n_actions).cuda()
        #self.val_fc = nn.Linear(LSTM_output_dim, head_hidden_dim).cuda()
        #self.val_out = nn.Linear(head_hidden_dim, 1).cuda()


    def forward(self, x:State) ->float:
        player = (x.dealer+len(x.bidding_sequence))%4
        cnt = (player-x.dealer+4)%4
        single_bid = torch.zeros(183, dtype=torch.float32)
        single_bid[:66] = x.features[player]
        tensor_list  = []
        while cnt <= len(x.bidding_sequence):
            _row = single_bid.clone()
            for i in range(3, 0, -1):
                if cnt - i < 0:
                    _row[66+(3-i)*39] = 1
                else:
                    #print(f"I: {i}, x.bidding_sequence[cnt-i]: {x.bidding_sequence[cnt-i]}")
                    _row[67+(3-i)*39+x.bidding_sequence[cnt-i]] = 1
            cnt+=4
            tensor_list.append(_row)
        #self.log.debug(f"tensor_list: {tensor_list}")
        tensor_list = torch.stack(tensor_list).unsqueeze(0).cuda()
        shared_out, _ = self.lstm(tensor_list.cuda())
        shared_out = shared_out[:, -1, :]
        shared_out =self.fc(shared_out)
        
        #adv = nn.ReLU()(self.adv_fc(shared_out))
        _adv = self.adv(shared_out)
        #print(f"adv: {adv}")
        #val = nn.ReLU()(self.val_fc(shared_out))
        _val = self.val(shared_out)
        #print(f"val: {val}")
        #print(f"mean: {torch.mean(adv)}")
        #q_values = (val-torch.mean(adv)) + adv
        q_values = _val + _adv - _adv.mean(dim=1, keepdim=True)
        #tba, adjustment for Q-value calc
        return q_values.flatten()
    
