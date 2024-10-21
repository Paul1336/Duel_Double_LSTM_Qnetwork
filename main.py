import torch
import collections
import random
import torch.optim as optim
from agent import DuelDDQNAgent
from model import Duel_DDNQ
from env import Env, State
from typing import List

# def create_b(self)-> B:
class EpsilonScheduler():
    epsilon = None
    min_val = None
    decay_rate = None
    def __init__(self, initial_val:float=1, min_val:float=0.01, decay_rate:float=0.995):
        self.epsilon = initial_val
        self.min_val = min_val
        self.decay_rate = decay_rate

    def update(self, episode:int):
        self.epsilon = max(self.min_val, self.epsilon * self.decay_rate)
        # tba

class Experiance():
    somethint: str# tba
class ReplayMemory():
    buffer:List[Experiance] = None
    def __init__(self, max_size:int):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp:Experiance):
        self.buffer.append(exp)

    def sample(self)-> Experiance:
        experience = random.sample(self.buffer, 1)
        state, action, reward, next_state, terminated = experience
        return state, action, reward, next_state, terminated

    def __len__(self)->int:
        return len(self.buffer)


# Hyperparameters
NUM_EPISODES = 500  # max ep
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
MIN_EPSILON = 0.01
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
MIN_MEMORY_SIZE = 64
DECAY_RATE = 0.995

env = Env()
pretrained_model = torch.load('pretrained_model.pth')
# import pretrain LSTM here
Q_model = Duel_DDNQ(pretrained_model).cuda()
optimizer = optim.Adam(Q_model.parameters(), lr=LEARNING_RATE)
agent = DuelDDQNAgent(Q_model, optimizer, DISCOUNT_FACTOR)
memory = ReplayMemory(MEMORY_SIZE)
epsilon_scheduler = EpsilonScheduler(initial_val=1.0, min_val=MIN_EPSILON, decay_rate=DECAY_RATE)


def train_agent():
    while len(memory) < MIN_MEMORY_SIZE:
        state = env.reset()
        terminated = False
        total_reward = 0
        while not terminated:
            action = agent.choose_action(state, EpsilonScheduler.epsilon)
            next_state, reward, terminated = env.step(action)
            memory.append((state, action, reward, next_state, terminated))
            state = next_state
            total_reward += reward

    for episode in range(NUM_EPISODES):
        epsilon_scheduler.update(episode)
        state = env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            action = agent.choose_action(state, EpsilonScheduler.epsilon)
            next_state, reward, terminated = env.step(action)
            memory.append((state, action, reward, next_state, terminated))

            if len(memory) >= MIN_MEMORY_SIZE:
                state, action, reward, next_state, terminated = memory.sample()
                agent.train(state, action, reward, next_state, terminated)

            state = next_state
            total_reward += reward

        if episode % TARGET_UPDATE == 0:
            agent.synchronous_networks()

        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward}")


if __name__ == "__main__":
    train_agent()