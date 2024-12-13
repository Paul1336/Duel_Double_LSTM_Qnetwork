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

    def log(self):
        log = logger.get_logger(__name__)
        position = ["North", "East", "South", "West"]
        for i in range(0, 4):
            log.debug(f"{position[i]}: ")
            log.debug(f"S  :{self.features[i][:13]}")
            log.debug(f"H  :{self.features[i][13:26]}")
            log.debug(f"D  :{self.features[i][26:39]}")
            log.debug(f"C  :{self.features[i][39:52]}")
            log.debug(f"HCP:{self.features[i][52:57]}")
            log.debug(f"len:{self.features[i][57:61]}")
            log.debug(f"bal:{self.features[i][61]}")
            log.debug(f"vul:{self.features[i][62:66]}")
        log.debug(f"features sequence: {self.bidding_sequence}")
        log.debug(f"last doubled: {self.last_doubled}")
        log.debug(f"last bid: {self.last_bid}")
        log.debug(f"last pass: {self.last_pass}")
        log.debug(f"dealer: {self.dealer}")
        log.debug(f"agent team: {self.agent_team}")