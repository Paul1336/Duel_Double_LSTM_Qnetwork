from model import *
from dealer import *
import torch

class RandomLSTM(nn.Module):
    def __init__(self, input_dim=183, hidden_dim=64, output_dim=38, num_layers=1):
        super(RandomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

def _test():
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
    else:
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    random_lstm = RandomLSTM()
    model = Duel_DDNQ(pretrained_model=random_lstm)
    deal = Dealer.new_game()
    state = deal.new_state
    with torch.no_grad():
        output = model(state)
        print(output.tolist())#38 classes

if __name__ == "__main__":
    _test()