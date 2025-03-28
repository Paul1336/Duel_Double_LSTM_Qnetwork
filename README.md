# The research record and pretrained LSTM source code

You can find the pretrained model source code at the following link:[https://github.com/Paul1336/Contract_Bridge_LSTM](https://github.com/Paul1336/Contract_Bridge_LSTM)

The dataset and Experiment Log:
[https://drive.google.com/drive/folders/1o8Xu1InRWeP3vMJ3RrRQa7mrjhiGmpHE?usp=drive_link](https://drive.google.com/drive/folders/1o8Xu1InRWeP3vMJ3RrRQa7mrjhiGmpHE?usp=drive_link)

# Data Structure and Encoding

- **`State`**:

  - `features[4]` model input hand feature field for 4 player
  - `bidding_sequence` a list, the large index implied the later bidding, convertion is neede for being model input bidding sequence feature field
  - `last_doubled` int, 0 denote no doubled before, -ith to be the last double/redouble otherwise
  - `last_bid` int, 0 denote no bid before, -ith to be the last bid otherwise
  - `last_pass` int, 0 denote no pass before, -ith to be the last pass otherwise
  - `dealer` int
  - `agent_team` int, 0: N/S, 1:E/W

- **`Deal`**

  - `new_state` State, for env
  - `vul` int[2], for reward.py and dll, vul[0] if NS vulnerable, vul[1] if EW
  - `pbn` str, "< North Space suit >.< North Heart suit >.< North Diamond suit >.< North Club suit > ES.EH.ED.EC ... ..."

- **`Dealer`**: N-0, E-1, S-2, W-3

- **`Suit-card`**: S, H, D, C
- **`Suit-bid/feature`**: C, D, H, S, NT
- **`action/bid`** int, 0 = p, 1 = d, 2 = r, 3 = 1c...37 = 7N

- **`level`**: actual level -1

- **`USE_CARD_DETAIL`**: if feature fields have cards info()

# Note

the current structure of reward recalculate par score and dd answer every time we call the imp_diff() in reward.py, if once the scoring value for q change to revaluatio each step instead of the teminated state, the method imp_diff() should be optimized.

output log stored here: https://drive.google.com/drive/folders/1_z0wdUCKif2fJrmsY-YzTDdZWEjdyBg3?usp=sharing

to compile .so file:
g++ -shared -fPIC -o linux_dds.so DoubleDummySolver.cpp -L. -ldds
ldd linux_dds.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
sudo apt-get update
sudo apt-get install libboost-thread-dev
sudo ln -s /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.83.0 /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.74.0

to evaluate:
python eval.py --path <path_to_pretrained_model.pt>
