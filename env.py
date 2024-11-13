from dataclasses import dataclass
import torch
from typing import Tuple
from dealer import Dealer, Deal
from reward import RewardCalculator, ddResponse
import random

@dataclass
class State:
    features: torch.tensor = None
    bidding_sequence: torch.tensor = None
    last_doubled: int
    last_bid: int
    last_pass: int
    dealer: int

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
    def random_action(state: State)->int:
        if state.last_bid == 0:
            bid = random.randint(3, 38)
            return bid % 38
        elif state.last_doubled < state.last_bid:
            if state.last_doubled == 2 or state.bidding_sequence[-state.last_doubled].item() == 2:
                bid = random.randint(state.bidding_sequence[-state.last_bid].item()+1, 38)
                return bid % 38
            else:
                bid = random.randint(state.bidding_sequence[-state.last_bid].item()+1, 39)
                if bid == 39:
                    bid = 40
                return bid % 38
        else:
            if state.last_bid == 2:
                bid = random.randint(state.bidding_sequence[-state.last_bid].item()+1, 38)
                return bid % 38
            else:
                bid = random.randint(state.bidding_sequence[-state.last_bid].item()+1, 39)
                return bid % 38


    def reset (self)-> State:
        deal = Dealer.new_game()
        self.current_state = deal.new_state
        self.reward_calculater = (RewardCalculator(deal.vul, deal.pbn))

    def step (self, action:int)-> Tuple[State, float, int]:# next_state, reward, terminate
        print("initialize")
        #tba, calc reward with reward.imploss() method and predict next state with qnetwork
        #tba, judge terminated

    def update_networks(self, Qnetwork):
        self.Qnetwork = Qnetwork