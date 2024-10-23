# import learning lib
import torch
import torch.optim as optim
# util lib
import collections
import random
from typing import List, Optional
# my customize lib
from agent import DuelDDQNAgent
from model import Duel_DDNQ
from env import Env, Experiance

# parameters
NUM_EPISODES = 500  # max ep
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
MIN_EPSILON = 0.01
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
MIN_MEMORY_SIZE = 64
DECAY_RATE = 0.995
HEAD_HIDDEN_DIM = 128
PRETRAINED_MODEL_PATH = 'pretrained_model.pth'# tba, save pretrained LSTM with pytorch method: torch.save

class EpsilonScheduler():
    epsilon:float = None
    min_val:float = None
    decay_rate:float = None

    def __init__(self, initial_val:float=1, min_val:float=0.01, decay_rate:float=0.995):
        self.epsilon = initial_val
        self.min_val = min_val
        self.decay_rate = decay_rate

    def update(self, episode:int):
        self.epsilon = max(self.min_val, self.epsilon * self.decay_rate)
        # tba, decay algo base on episode or other factor


class ReplayMemory():
    buffer:Optional[Experiance] = None
    def __init__(self, max_size:int):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp:Experiance):
        self.buffer.append(exp) # discard the earliest obj and free memory

    def sample(self)-> Experiance:
        experience = random.sample(self.buffer, 1)
        return experience

    def __len__(self)->int:
        return len(self.buffer)


def train_agent():
    pretrained_model = torch.load(PRETRAINED_MODEL_PATH)
    Q_model = Duel_DDNQ(pretrained_model, HEAD_HIDDEN_DIM).cuda()
    optimizer = optim.Adam(Q_model.parameters(), lr=LEARNING_RATE)
    agent = DuelDDQNAgent(Q_model, optimizer, DISCOUNT_FACTOR)
    memory = ReplayMemory(MEMORY_SIZE)
    env = Env(agent.get_network())
    epsilon_scheduler = EpsilonScheduler(initial_val=1.0, min_val=MIN_EPSILON, decay_rate=DECAY_RATE)
    
    while len(memory) < MIN_MEMORY_SIZE:
        state = env.reset()
        terminated = False
        while not terminated:
            action = agent.choose_action(state, EpsilonScheduler.epsilon)
            next_state, reward, terminated = env.step(action)
            memory.append(Experiance(state, action, reward, next_state, terminated))
            state = next_state

    for episode in range(NUM_EPISODES):
        epsilon_scheduler.update(episode)
        state = env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            action = agent.choose_action(state, EpsilonScheduler.epsilon)
            next_state, reward, terminated = env.step(action)
            memory.append(Experiance(state, action, reward, next_state, terminated))
            state = next_state
            total_reward += reward

            if len(memory) >= MIN_MEMORY_SIZE:
                exp = memory.sample()
                agent.train(exp)

        if episode % TARGET_UPDATE == 0:
            agent.synchronous_networks()
        env.update_networks(agent.get_network())
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward}")
        #tba, loggin and saving model
        #tba, contral during process

if __name__ == "__main__":
    train_agent()