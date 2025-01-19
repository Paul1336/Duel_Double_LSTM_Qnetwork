import ctypes
import platform
from dataclasses import dataclass
from state import State
import logger


@dataclass
class RewardCalculator:
    pbn_str: str
    vul: list[int]

    def __init__ (self, _deal: str, _vul:list[int]):
        pass

    def imp_diff (self, state:State) -> float:
        return -1
      