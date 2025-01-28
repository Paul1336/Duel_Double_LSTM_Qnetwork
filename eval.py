import argparse
import torch
import torch.nn as nn
from env import Env
from state import State
import logger
from dealer import Deal, Dealer
from reward import RewardCalculator

log = logger.get_logger(__name__)
class Duel_DDNQ(nn.Module):
    def __init__(self, pretrained_model_state_dict, LSTM_output_dim=38, head_hidden_dim=128):
        super(Duel_DDNQ, self).__init__()

        # initialize LSTM layer
        self.lstm = nn.LSTM(
            input_size=183,
            hidden_size=1024,
            num_layers=4,
            batch_first=True
        )

        lstm_state_dict = {
            key.replace("lstm1.", ""): value
            for key, value in pretrained_model_state_dict.items()
            if key.startswith("lstm1.")
        }
        self.lstm.load_state_dict(lstm_state_dict)
        self.lstm = self.lstm.cuda()

        # initialize fc layer
        self.fc = nn.Linear(1024, LSTM_output_dim)
        self.fc.weight.data = pretrained_model_state_dict["fc.weight"].clone()
        self.fc.bias.data = pretrained_model_state_dict["fc.bias"].clone()
        self.fc = self.fc.cuda()

        # initialize Advantage head
        self.adv_fc = nn.Linear(LSTM_output_dim, head_hidden_dim)
        self.adv_fc.weight.data = pretrained_model_state_dict["adv_fc.weight"].clone()
        self.adv_fc.bias.data = pretrained_model_state_dict["adv_fc.bias"].clone()

        self.adv_out = nn.Linear(head_hidden_dim, Env.n_actions)
        self.adv_out.weight.data = pretrained_model_state_dict["adv_out.weight"].clone()
        self.adv_out.bias.data = pretrained_model_state_dict["adv_out.bias"].clone()

        # initialize Value head
        self.val_fc = nn.Linear(LSTM_output_dim, head_hidden_dim)
        self.val_fc.weight.data = pretrained_model_state_dict["val_fc.weight"].clone()
        self.val_fc.bias.data = pretrained_model_state_dict["val_fc.bias"].clone()

        self.val_out = nn.Linear(head_hidden_dim, 1)
        self.val_out.weight.data = pretrained_model_state_dict["val_out.weight"].clone()
        self.val_out.bias.data = pretrained_model_state_dict["val_out.bias"].clone()

        self.adv_fc = self.adv_fc.cuda()
        self.adv_out = self.adv_out.cuda()
        self.val_fc = self.val_fc.cuda()
        self.val_out = self.val_out.cuda()

    def forward(self, x:State):
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
        adv = nn.ReLU()(self.adv_fc(shared_out))
        adv = self.adv_out(adv)
        #print(f"adv: {adv}")
        val = nn.ReLU()(self.val_fc(shared_out))
        val = self.val_out(val)
        #print(f"val: {val}")
        #print(f"mean: {torch.mean(adv)}")
        q_values = (val-torch.mean(adv)) + adv
        #tba, adjustment for Q-value calc
        return q_values.flatten()


def load_model(path):
    try:
        checkpoint = torch.load(path)
        print(f"Model loaded successfully from {path}")
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load the model from {path}: {e}")

def log_game(state:State, pbn:str, action:int = -1):
    if action == -1:
        log.info(f"initial info: {state}")
    else:
        vul_str = ""
        action_str = ""
        if state.features[(state.dealer+len(state.bidding_sequence)-1)%4][62].item() == 1:
            vul_str = "self"
        elif state.features[(state.dealer+len(state.bidding_sequence)-1)%4][63].item() == 1:
            vul_str = "both"
        elif state.features[(state.dealer+len(state.bidding_sequence)-1)%4][64].item() == 1:
            vul_str = "none"
        else:
            vul_str = "opp"
        if action == 0:
            action_str += "P"
        elif action == 1:
            action_str += "D"
        elif action == 2:
            action_str += "R"
        else:
            action_str += str(1+(action-3)//5)
            if (action-3)%5 == 0:
                action_str += "C"
            elif (action-3)%5 == 1:
                action_str += "D"
            elif (action-3)%5 == 2:
                action_str += "H"
            elif (action-3)%5 == 3:
                action_str += "S"
            elif (action-3)%5 == 4:
                action_str += "N"
        hand = pbn.split("N:")[1].split()[(state.dealer+len(state.bidding_sequence)-1)%4]
        log.info(f"hands: {hand}")
        log.info(f"vul: {vul_str}")
        log.info(f"action: {action_str}")
        log.info(f"bidding sequence: {state.bidding_sequence}\n")

def random_test(model):
    env = Env()
    state, pbn = env.reset()
    turn = (state.dealer) % 2
    terminated = False
    reward = 0
    log_game(state, pbn, -1)
    while terminated != 1:
        log.debug("choosing action(for memory gen)")
        actions = model(state)
        action = torch.argmax(actions).item()
        log.debug("action selected")
        #print(f"env.step(action): {action}")
        next_state, reward, terminated = env.step(action)
        log_game(next_state, pbn, action)
        turn = (turn+1)%2
        state = next_state

    if turn%2 == 1:
        log.info(f"Total Reward, alphe: {reward}, beta: {-reward}")
    else:
        log.info(f"Total Reward, alphe: {-reward}, beta: {reward}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained Duel_DDNQ model.")
    parser.add_argument("--path", type=str, required=True, help="Path to the pretrained model (.pt file)")
    args = parser.parse_args()

    pretrained_state_dict = load_model(args.path)

    model = Duel_DDNQ(pretrained_model_state_dict=pretrained_state_dict)
    model.eval()
    while True:
        cmd = input("option: (r=random deal test, s=specified state, q=quit): ").strip()
        if cmd == 'r':
            random_test(model)
        elif cmd == 's':
            pass
        else:
            break




if __name__ == "__main__":
    main()
