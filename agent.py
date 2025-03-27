import torch
import numpy as np
import copy
from env import Env, State, Experiance
import logger

class DuelDDQNAgent():
    Q_main = None
    Q_target = None
    optimizer = None
    discount_factor = None

    def __init__(self, Q_model, optimizer, discount_factor:float=0.99, temperature: float = 1):
        self.Q_main = copy.deepcopy(Q_model)
        self.Q_origin = copy.deepcopy(Q_model)
        self.Q_target = copy.deepcopy(Q_model)
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.log = logger.get_logger(__name__)


    def choose_action_origin(self, state: State, epsilon: float) ->int:
        self.log.debug("pretrained action\n")
        with torch.no_grad():
            legal_actions = Env.action_space(state)
            filtered_values = self.pred_model(state, "origin")[legal_actions]
            return legal_actions[torch.argmax(filtered_values).item()]

    
    def choose_action(self, state: State, epsilon: float) ->int:
        #print("choosing action:\n")
        #print("epsilon = ", epsilon)
        self.log.debug(f"epsilon = {epsilon}\n")
        if np.random.uniform(0, 1) < epsilon:
            legal_actions = Env.action_space(state)
            self.log.debug(f"legal_actions = {legal_actions}\n")
            chosen_index = np.random.choice(len(legal_actions))
            return legal_actions[chosen_index]
            #filtered_values = self.pred_model(state, "main")[legal_actions]
            #filtered_values = filtered_values - torch.max(filtered_values)
            #self.log.debug(f"filtered_values = {filtered_values}\n")
            #probabilities = torch.softmax(filtered_values / self.temperature, dim=0)
            #self.log.debug(f"probabilities = {probabilities}\n")
            #chosen_index = torch.multinomial(probabilities, num_samples=1)
            #return legal_actions[chosen_index.item()]
            #return Env.random_action(state)
        else:
            self.log.debug("predicted action\n")
            with torch.no_grad():
                legal_actions = Env.action_space(state)
                filtered_values = self.pred_model(state, "main")[legal_actions]
                return legal_actions[torch.argmax(filtered_values).item()]
                #return torch.argmax(self.pred_model(state, "main")).item()
            
    def synchronous_networks(self):
        self.Q_target = copy.deepcopy(self.Q_main)

    def pred_model(self, state, target) -> State:
        if target == "target":
            return self.Q_target(state)
        elif target == "main":
            return self.Q_main(state)
        else:
            return self.Q_origin(state)
                
    def train(self, exp:Experiance):
        #Update the agent.
        #print(f"exp len: {len(exp)}")
        with torch.no_grad():
            next_q_values_target= self.pred_model(exp.next_state, "target")
            next_q_values_main = self.pred_model(exp.next_state, "main")
        next_action = torch.argmax(next_q_values_main).item()
        #next_q_values_main.argmax(dim=1).item()
        
        #q_values = torch.max(self.pred_model(exp.state, "main"))
        q_values = torch.max(self.pred_model(exp.state, "main"))
        next_q_value = next_q_values_target[next_action]
        target_q_value = exp.reward + (1 - exp.terminated) * self.discount_factor * next_q_value
        #print(f"target_q_value: {target_q_value}")
        #print(f"q_values: {q_values}")
        loss = target_q_value-q_values
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def log(self):
        pass

    def save_model(self, path):
        try:
            torch.save(self.Q_main.state_dict(), path)
            self.log.info(f"Model saved successfully to {path}")
        except Exception as e:
            raise RuntimeError(f"agent.save_model() Failed to save model to {path}: {e}") from e
