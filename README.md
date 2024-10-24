# Project Overview

## (WIP)

- **`main.py`**:
  - `EpsilonScheduler.update()` decay algo base on episode or other factor
  - `parameters` save pretrained LSTM with pytorch method: torch.save

- **`model.py`**:
  - `__init__()` adjustment for the model's heads
  - `forward()` convert input from state type to tensor
  - `forward()` adjustment for Q-value calc

## (TBA)

- **`main.py`**:
  - `main()` loggin and saving model
  - `main()` contral during process

- **`env.py`**:
  - `random_action()` random return an action(int) depends on current state
  - `step()` calc reward and predict next state with qnetwork
  - `step()` judge terminated

- **`reward.py`**:
  - `imp_diff()` record dealer of suit and return loss

- **`dealer.py`**:
  - `new_game()`

## Pretrained Model Source Code

You can find the pretrained model source code at the following link:  
[Pretrained Model Source Code](https://github.com/Paul1336/Contract_Bridge_LSTM)

## Data Structure and Encoding

- **`State`**:
  - `features` model input hand feature field
  - `bidding_sequence` a list, the large index implied the later bidding, convertion is neede for being model input bidding sequence feature field

- **`State`**: int, 0 < x < 37 , denote p, d, r, 1c...7s, 7n

- **`Deal`**
  - `new_state` State, for env
  - `vul` int[2], for reward.py and dll, vul[0] if NS vulnerable, vul[1] if EW
  - `pbn` str, "< North Space suit >.< North Heart suit >.< North Diamond suit >.< North Club suit > ES.EH.ED.EC ... ..."

- **`Dealer`**: N-0, E-1, S-2, W-3

- **`Suit`**: C-0, D-1, H-2, S-3, NT-4
