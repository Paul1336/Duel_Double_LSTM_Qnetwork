from dataclasses import dataclass
import torch
from typing import Tuple
from dealer import Dealer
from state import State
from reward import RewardCalculator
import random


@dataclass
class Experiance():
    episode: int
    state: State
    action: int
    reward: int
    next_state: State
    terminated: int
    def log(self, f):
        f.write("curent state: \n")
        f.write(f"episode generated: {self.episode}\n")
        self.state.log(f)
        f.write(f"action: {self.action}\n")
        f.write(f"reward: {self.reward}\n")
        f.write("next state: \n")
        self.next_state.log(f)
        f.write(f"terminated: {self.terminated}\n")

class Env():
    n_actions:int = 38
    current_state:State = None
    reward_calculater = None
    def __init__ (self):
        self.reset()

    def reset (self)-> State:
        new_game = Dealer.new_game()
        self.current_state = new_game.new_state
        self.reward_calculater = (RewardCalculator(new_game.pbn, new_game.vul))

    @staticmethod
    def action_space(state: State)->list:
        #if state.last_bid == 1:
        #    print(f"last bid {state.last_bid}, val: {state.bidding_sequence[-state.last_bid]}")
        if state.last_bid == 0 and state.last_doubled == 0:
            return list({0} | set(range(3, 37)))
        elif state.last_doubled < state.last_bid and state.last_doubled != 0:
            if state.last_doubled == 2 or state.bidding_sequence[-state.last_doubled] == 2:
                return list({0} | set(range(state.bidding_sequence[-state.last_bid]+1, 37)))
            else:
                return list({0, 2} | set(range(state.bidding_sequence[-state.last_bid]+1, 37)))
        else:
            if state.last_bid == 2:
                return list({0} | set(range(state.bidding_sequence[-state.last_bid]+1, 37)))
            else:
                return list({0, 1} | set(range(state.bidding_sequence[-state.last_bid]+1, 37)))

    @staticmethod
    def random_action(state: State)->int:
        #print(f"choices: {Env.action_space(state)}")
        return random.choice(Env.action_space(state))

    def step (self, action:int)-> Tuple[State, float, int]:# next_state, reward, terminate
        _terminated = 0
        _reward = 0
        #print(self.current_state.bidding_sequence)
        #print(torch.cat((self.current_state.bidding_sequence, torch.tensor([action], dtype=torch.int64))))
        self.current_state.bidding_sequence.append(action)
        #self.current_state.bidding_sequence = torch.cat((self.current_state.bidding_sequence, torch.tensor([action], dtype=torch.int64)))
        if self.current_state.last_bid > 0:
            self.current_state.last_bid += 1
        if self.current_state.last_doubled > 0:
            self.current_state.last_doubled+=1
        if self.current_state.last_pass > 0:
            self.current_state.last_pass+=1
        _terminated = 0
        if action>2:
            self.current_state.last_bid = 1
        elif action>0:
            self.current_state.last_doubled = 1
        else:
            self.current_state.last_pass = 1
            #print(len(self.current_state.bidding_sequence))
            if (len(self.current_state.bidding_sequence) == 4 and self.current_state.last_bid == 0) or (self.current_state.last_bid > 3 and (self.current_state.last_doubled > 3 or self.current_state.last_doubled == 0)):
                _terminated = 1
                _reward = self.reward_calculater.imp_diff(self.current_state)   
        return self.current_state, _reward, _terminated   

    def log(self):
        pass