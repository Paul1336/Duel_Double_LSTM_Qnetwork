import torch
from env import Env

class DuelDDQNAgent():
    def __init__(self,
                 Q_model,
                 optimizer, 
                 discount_factor=0.99,
                 ):
        # objects
        self.Q_main = Q_model
        self.Q_target = Q_model
        self.optimizer = optimizer
        # hyperparameters
        self.discount_factor = discount_factor
        # private

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return Env.random_action(state)
        else:
            with torch.no_grad():
                return self.pred_model(state, "main")
            
    def synchronous_networks(self):
        self.Q_target = self.Q_main

    def pred_model(state, target):
        if target == "target":
            return self.Q_target(state)
        else:
            return self.Q_main(state)
                
    def train(self, state, action, reward, next_state, terminated):
        """Update the agent."""        
        with torch.no_grad():
            next_q_values_target= self.pred_model(next_state, "target")
            next_q_values_main = self.pred_model(next_state, "main")
        next_action = next_q_values_main.argmax(dim=1).item()
        
        q_values = self.pred_model(state, "main")
        next_q_value = next_q_values_target[next_action]
        target_q_value = reward + (1 - terminated) * self.discount_factor * next_q_value
        
        loss = target_q_value-q_values
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()