from dataclasses import dataclass

@dataclass
class State:
    var1: str# declare something
#action: int

class Env():
    action_space = ""
    current_state = ""
    def __init__ (self):
        print("initialize")# initialize, do something

    @classmethod
    def random_action(cls, state):
        return cls.action_space

    def reset (self):
        print("initialize")# do something

    def step (self):
        print("initialize")# do something