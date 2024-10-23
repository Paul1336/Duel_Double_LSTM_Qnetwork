from dataclasses import dataclass
import torch

@dataclass
class State:
    features: torch.tensor = None
    bidding_sequence: torch.tensor = None

@dataclass
class Deal:
    new_state: State = None
    vul: list[int] = None
    pbn: str = None

class Dealer():
    @classmethod
    def new_game()->Deal:
        return Deal({})
        #tba