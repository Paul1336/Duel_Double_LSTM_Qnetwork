# import learning lib
import torch
import torch.optim as optim
# util lib
import collections
import random
import math
import logger
import signal
import os
# my customize lib
from agent import DuelDDQNAgent
from model import Duel_DDNQ
from env import Env, Experiance

# parameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
MIN_MEMORY_SIZE = 64

PRETRAINED_MODEL_PATH = './pretrained_LSTM.pt'
# TBA, save pretrained LSTM(full feature) with pytorch method: torch.save

class EpsilonScheduler():
    epsilon:float = None
    start:float = None
    end:float = None
    decay_rate:float = None

    def __init__(self, initial_val:float=1, min_val:float=0.01, decay_rate:float=1000):
        self.epsilon = initial_val
        self.start = initial_val
        self.end = min_val
        self.decay_rate = decay_rate

    def update(self, episode:int):
        try:
            self.epsilon = self.end + (self.start - self.end) * math.exp(-1. * episode / self.decay_rate)
        except:
            raise RuntimeError(f"EpsilonScheduler.update() occur an error: {e}") from e

    def log(self):
        log.debug(f"Epsilon Scheduler State:")
        log.debug(f"Current Epsilon: {self.epsilon}")
        log.debug(f"Start Value: {self.start}")
        log.debug(f"End Value: {self.end}")
        log.debug(f"Decay Rate: {self.decay_rate}")

class ReplayMemory():
    buffer: collections.deque[Experiance]
    max_size = None
    def __init__(self, max_size:int):
        self.buffer = collections.deque(maxlen=max_size)
        self.max_size = max_size

    def append(self, exp:Experiance):
        try:
            self.buffer.append(exp)
        except:
            raise RuntimeError(f"ReplayMemory.append() occur an error: {e}") from e

    def sample(self)-> Experiance:
        try:
            experience = random.sample(self.buffer, 1)
        except:
            raise RuntimeError(f"ReplayMemory.sample() occur an error: {e}") from e
        return experience

    def __len__(self)->int:
        return len(self.buffer)

    def log(self):
        if not self.buffer:
            log.debug("No experiences to log.")
        else:
            log.debug(f"Replay Memory State:")
            log.debug(f"Current Buffer Size: {len(self.buffer)}")
            log.debug(f"Max Buffer Size: {self.max_size}")
            log.debug("All Experiences:")
            for i, exp in enumerate(self.buffer, 1):
                log.debug(f"--- Experience {i} ---")
                exp.log()


def train_agent(episode):
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
    if (episode+1) % TARGET_UPDATE == 0:
        agent.synchronous_networks()
    env.update_networks(agent.get_network())
    log.info(f"Episode: {episode+1}, Total Reward: {total_reward}")


def build_memory(max_size, min_size):
    memory = ReplayMemory(max_size)
    while len(memory) < min_size:
        state = env.reset()
        terminated = False
        while not terminated:
            action = agent.choose_action(state, EpsilonScheduler.epsilon)
            next_state, reward, terminated = env.step(action)
            memory.append(Experiance(state, action, reward, next_state, terminated))
            state = next_state
    return memory

def log_process_state():
    log.info(f"log process states: ")
    agent.log()
    env.log()
    epsilon_scheduler.log()
    memory.log()

def option_handler(signum, frame):
    global running
    running = False
    log.info("process paused")

signal.signal(signal.SIGUSR1, option_handler)

if __name__ == "__main__":
    log = logger.get_logger(__name__)
    log.info("starting process")
    pid = os.getpid()
    log.info(f"running on process ID = {pid}")
    print(f"use command: \"kill -SIGUSR1 {pid}\" to go to the menu")
    print("initialize...")
    pretrained_model = torch.load(PRETRAINED_MODEL_PATH)
    #TBA
    log.info(f"load pretrained model: {PRETRAINED_MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device = {device}")
    Q_model = Duel_DDNQ(pretrained_model).to(device)
    log.info(f"initialize q network")
    optimizer = optim.Adam(Q_model.parameters(), lr=LEARNING_RATE)
    log.info(f"initialize adam optimizer, learning rate = {LEARNING_RATE}")
    agent = DuelDDQNAgent(Q_model, optimizer, DISCOUNT_FACTOR)
    log.info(f"build agent, discount factor = {DISCOUNT_FACTOR}, target q update per {TARGET_UPDATE} epochs")
    env = Env(agent.get_network())
    log.info(f"build enviroment")
    epsilon_scheduler = EpsilonScheduler(initial_val=EPS_START, min_val=EPS_END, decay_rate=EPS_DECAY)
    log.info(f"build scheduler, initial value = {EPS_START}, minimun value = {EPS_END}, decay rate = {EPS_DECAY}")
    memory = build_memory(MEMORY_SIZE, MIN_MEMORY_SIZE)
    log.info(f"build replay memory including experience {MIN_MEMORY_SIZE}, max size {MEMORY_SIZE}")
    print(f"project initialized, start training agent...")
    running = True
    run_forever = True
    steps_to_run = 0
    episode = 0
    while True:
        if running:
            # training loop
            try:
                if run_forever:
                    train_agent(episode)
                    episode+=1
                else:
                    if steps_to_run > 0:
                        train_agent(episode)
                        episode+=1
                        steps_to_run -= 1
                    else:
                        running = False
                        log.info("specified training loop finished")
            except Exception as e:
                log.error(e)
                running = False

        else:
            cmd = input("process paused, option: (r=resume, rN=resume N times, l=log current states, s=save model, q=quit): ").strip()
            if cmd == 'r':
                running = True
                run_forever = True
                steps_to_run = 0
                log.info("process resumed")
            elif cmd.startswith('r') and cmd[1:].isdigit():
                n = int(cmd[1:])
                running = True
                run_forever = False
                steps_to_run = n
                log.info(f"process resumed for {n} steps")
            elif cmd == 'l':
                log_process_state()                
            elif cmd == 's':
                agent.save_model()  
                break
            elif cmd == 'q':
                log.info("process terminated")
                break
            else: