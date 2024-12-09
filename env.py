from dataclasses import dataclass
import torch
from typing import Tuple
from dealer import Dealer
from state import State
from reward import RewardCalculator, ddResponse
import random


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
            return random.choice(list({0} | set(range(3, 37))))
        elif state.last_doubled < state.last_bid:
            if state.last_doubled == 2 or state.bidding_sequence[-state.last_doubled] == 2:
                return random.choice(list({0} | set(range(state.bidding_sequence[-state.last_bid]+1, 37))))
            else:
                return random.choice(list({0, 2} | set(range(state.bidding_sequence[-state.last_bid]+1, 37))))
        else:
            if state.last_bid == 2:
                return random.choice(list({0} | set(range(state.bidding_sequence[-state.last_bid]+1, 37))))
            else:
                return random.choice(list({0, 1} | set(range(state.bidding_sequence[-state.last_bid]+1, 37))))

    def predict (self)->int:
        _prediction = self.Qnetwork(self.current_state)
        state = self.current_state
        legel_bids = []
        if state.last_bid == 0:
            legel_bids = list({0} | set(range(3, 37)))
        elif state.last_doubled < state.last_bid:
            if state.last_doubled == 2 or state.bidding_sequence[-state.last_doubled] == 2:
                legel_bids = list({0} | set(range(state.bidding_sequence[-state.last_bid]+1, 37)))
            else:
                legel_bids = list({0, 2} | set(range(state.bidding_sequence[-state.last_bid]+1, 37)))
        else:
            if state.last_bid == 2:
                legel_bids = list({0} | set(range(state.bidding_sequence[-state.last_bid]+1, 37)))
            else:
                legel_bids = list({0, 1} | set(range(state.bidding_sequence[-state.last_bid]+1, 37)))
        filtered_prediction = [_prediction[i] for i in legel_bids]
        return max(filtered_prediction)
    
    def reset (self)-> State:
        deal = Dealer.new_game()
        self.current_state = deal.new_state
        self.reward_calculater = (RewardCalculator(deal.vul, deal.pbn))
        if(self.current_state.dealer%4 == self.current_state.agent_team):
            self.record_bidding(self.predict())

    def record_bidding (self, action:int):
        self.current_state.bidding_sequence.append(action)
        self.current_state.last_bid += 1
        self.current_state.last_doubled+=1
        self.current_state.last_pass+=1
        _terminated = 0
        if action>2:
            self.current_state.last_bid = 1
        elif action>0:
            self.current_state.last_doubled = 1
        else:
            self.current_state.last_pass = 1
            if action == 0 and self.current_state.last_bid > 1 and self.current_state.last_doubled > 1:
                _terminated = 1
        return _terminated

    def step (self, action:int)-> Tuple[State, float, int]:# next_state, reward, terminate
        _terminated = 0
        _reward = 0
        if self.record_bidding(action) == 1:
            _terminated = 1
            _reward = self.reward_calculater.imp_diff(self.current_state)
        else:
            #pred
            if self.record_bidding(self.predict()) == 1:
                _terminated = 1
                _reward = self.reward_calculater.imp_diff(self.current_state)
        
        return self.current_state, _reward, _terminated
        #tba, calc reward with reward.imploss() method and predict next state with qnetwork

    def update_networks(self, Qnetwork):
        self.Qnetwork = Qnetwork