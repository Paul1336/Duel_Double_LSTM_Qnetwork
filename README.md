# The research record and pretrained LSTM source code

You can find the pretrained model source code at the following link:[https://github.com/Paul1336/Contract_Bridge_LSTM](https://github.com/Paul1336/Contract_Bridge_LSTM)

The dataset and Experiment Log:
[https://drive.google.com/drive/folders/1o8Xu1InRWeP3vMJ3RrRQa7mrjhiGmpHE?usp=drive_link](https://drive.google.com/drive/folders/1o8Xu1InRWeP3vMJ3RrRQa7mrjhiGmpHE?usp=drive_link)

# Project Overview

## (WIP)

- **`main.py`**:

  - `EpsilonScheduler.update()` decay algo base on episode or other factor V
  - `parameters` save pretrained LSTM with pytorch method: torch.save, try full data set w/ cards info

- **`model.py`**:
  - `__init__()` adjustment for the model's heads
  - `forward()` convert input from state type to tensor
  - `forward()` adjustment for Q-value calc

## (TBA)

- **`main.py`**:

  - `main()` loggin and saving model
  - `main()` contral during process

- **`env.py`**:

  - `random_action()` random return an action(int) depends on current state V
  - `step()` calc reward and predict next state with qnetwork V
  - `step()` judge terminated V

- **`reward.py`**:

  - `imp_diff()` the new version need to convert state to contract info

- **`DDS.cpp`**:

  - `ddAnalize()` the new version only return imp diff

- **`dealer.py`**:
  - `new_game()` V

## Data Structure and Encoding

- **`State`**:

  - `features[4]` model input hand feature field for 4 player
  - `bidding_sequence` a list, the large index implied the later bidding, convertion is neede for being model input bidding sequence feature field
  - `last_doubled` int, 0 denote no doubled before, -ith to be the last double/redouble otherwise
  - `last_bid` int, 0 denote no bid before, -ith to be the last bid otherwise
  - `last_pass` int, 0 denote no pass before, -ith to be the last pass otherwise
  - `dealer` int

- **`Deal`**

  - `new_state` State, for env
  - `vul` int[2], for reward.py and dll, vul[0] if NS vulnerable, vul[1] if EW
  - `pbn` str, "< North Space suit >.< North Heart suit >.< North Diamond suit >.< North Club suit > ES.EH.ED.EC ... ..."

- **`Dealer`**: N-0, E-1, S-2, W-3

- **`Suit`**: C-0, D-1, H-2, S-3, NT-4

- **`action/bid`** int, 0 = p, 1 = d, 2 = r, 3 = 1c...37 = 7N

- **`USE_CARD_DETAIL`**: if feature fields have cards info()
