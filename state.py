from dataclasses import dataclass, field
import torch
import logger


@dataclass
class State:
    features: torch.tensor = field(default_factory=lambda: torch.empty(0))
    bidding_sequence: list[int] = field(default_factory=list)
    last_doubled: int = 0
    last_bid: int = 0
    last_pass: int = 0
    dealer: int = 0
    agent_team: int = 0

    def log(self, f):
        position = ["North", "East", "South", "West"]
        for i in range(0, 4):
            f.write(f"{position[i]}: \n")
            f.write(f"S  :{self.features[i][:13]}\n")
            f.write(f"H  :{self.features[i][13:26]}\n")
            f.write(f"D  :{self.features[i][26:39]}\n")
            f.write(f"C  :{self.features[i][39:52]}\n")
            f.write(f"HCP:{self.features[i][52:57]}\n")
            f.write(f"len:{self.features[i][57:61]}\n")
            f.write(f"bal:{self.features[i][61]}\n")
            f.write(f"vul:{self.features[i][62:66]}\n")
        f.write(f"features sequence: {self.bidding_sequence}\n")
        f.write(f"last doubled: {self.last_doubled}\n")
        f.write(f"last bid: {self.last_bid}\n")
        f.write(f"last pass: {self.last_pass}\n")
        f.write(f"dealer: {self.dealer}\n")
        f.write(f"agent team: {self.agent_team}\n")