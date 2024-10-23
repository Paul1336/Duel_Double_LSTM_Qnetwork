from dataclasses import dataclass
import torch
from typing import Tuple
from dealer import Dealer, Deal
from reward import RewardCalculator, ddResponse

@dataclass
class State:
    features: torch.tensor = None
    bidding_sequence: torch.tensor = None

@dataclass
class Experiance():
    state: State
    action: int
    reward: float
    next_state: State
    terminated: int

class Env():
    n_actions:int = 38
    current_state:State = None
    reward_calculater = None
    Qnetwork = None
    def __init__ (self, Qnetwork):
        self.Qnetwork = Qnetwork
        self.reset()

    @classmethod
    def random_action(state)->int:
        return Env.action_space 
        # tba, depends on state

    def reset (self)-> State:
        deal = Dealer.new_game()
        current_state = deal.new_state
        self.reward_calculater = (RewardCalculator(deal.vul, deal.pbn))

    def step (self, action:int)-> Tuple[State, float, int]:# next_state, reward, terminate
        print("initialize")
        #tba, calc reward with reward.imploss() method and predict next state with qnetwork
        #tba, judge terminated

    def update_networks(self, Qnetwork):
        self.Qnetwork = Qnetwork