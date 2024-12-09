from dataclasses import dataclass, field
import torch


@dataclass
class State:
    features: torch.tensor = field(default_factory=lambda: torch.empty(0))
    bidding_sequence: list[int] = field(default_factory=list)
    last_doubled: int = 0
    last_bid: int = 0
    last_pass: int = 0
    dealer: int = 0
    agent_team: int = 0

    def print_state(self):
        print("features len:", len(self.features[0]))
        position = ["North", "East", "South", "West"]
        for i in range(0, 4):
            print(position[i], ":")
            print("S  :", self.features[i][:13])
            print("H  :", self.features[i][13:26])
            print("D  :", self.features[i][26:39])
            print("C  :", self.features[i][39:52])
            print("HCP:", self.features[i][52:57])
            print("len:", self.features[i][57:61])
            print("bal:", self.features[i][61])
            print("vul:", self.features[i][62:66])
        print("features sequence:", self.bidding_sequence)
        print("last doubled:", self.last_doubled)
        print("last bid:", self.last_bid)
        print("last pass:", self.last_pass)
        print("dealer:", self.dealer)
        print("agent team:", self.agent_team)