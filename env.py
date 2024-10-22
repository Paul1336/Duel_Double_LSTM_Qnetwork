from dataclasses import dataclass
import torch
from typing import Tuple

@dataclass
class State:
    features: torch.tensor = None
    bidding_sequence: torch.tensor = None
    #torch.cat, the last 3 is the previos tensor

#action: int(ont hot encoding bidding)

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
    imp_loss:ImpChart = None# tba
    Qnetwork = None
    def __init__ (self, Qnetwork):
        self.Qnetwork = Qnetwork
        self.reset()

    @classmethod
    def random_action(cls, state)->int:
        return cls.action_space#tba, depends on state

    def reset (self)-> State:
        print("initialize")
        #tba: deal, calc DDA and IMP

    def step (self)-> Tuple[State, float, int]:#next_state, reward, terminate
        print("initialize")#tba, predict next state w/ qnetwork

    def update_networks(self, Qnetwork):
        self.Qnetwork = Qnetwork