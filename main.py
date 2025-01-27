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
import copy
from datetime import datetime
import matplotlib.pyplot as plt
# my customize lib
from agent import DuelDDQNAgent
from model import Duel_DDNQ
from env import Env, Experiance
from state import State

# parameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 1000
CREATE_MEMORY_EPSILON = 0.5
TARGET_UPDATE = 100
MEMORY_SIZE = 10000
MIN_MEMORY_SIZE = 100
alpha_rewards = []
beta_rewards = []

PRETRAINED_MODEL_PATH = '../0125_bestmodel.pt'
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
    def __init__(self, max_size:int):
        self.buffer = collections.deque(maxlen=max_size)
        self.max_size = max_size

    def append(self, exp:Experiance):
        try:
            #print("exp: ")
            #print(f"episode: {exp.episode}")
            #print(f"state: {exp.state}")
            #print(f"action: {exp.action}")
            #print(f"reward: {exp.reward}")
            #print(f"next_state: {exp.next_state}")
            #print(f"terminated: {exp.terminated}")
            #print(f"====================================================")
            
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

def log_game(state:State, pbn:str, action:int = -1):
    if action == -1:
        log.debug(f"initial info: {state}")
    else:
        vul_str = ""
        action_str = ""
        if state.features[(state.dealer+len(state.bidding_sequence)-1)%4][62].item() == 1:
            vul_str = "self"
        elif state.features[(state.dealer+len(state.bidding_sequence)-1)%4][63].item() == 1:
            vul_str = "both"
        elif state.features[(state.dealer+len(state.bidding_sequence)-1)%4][64].item() == 1:
            vul_str = "none"
        else:
            vul_str = "opp"
        if action == 0:
            action_str += "P"
        elif action == 1:
            action_str += "D"
        elif action == 2:
            action_str += "R"
        else:
            action_str += str(1+(action-3)//5)
            if (action-3)%5 == 0:
                action_str += "C"
            elif (action-3)%5 == 1:
                action_str += "D"
            elif (action-3)%5 == 2:
                action_str += "H"
            elif (action-3)%5 == 3:
                action_str += "S"
            elif (action-3)%5 == 4:
                action_str += "N"
        hand = pbn.split("N:")[1].split()[(state.dealer+len(state.bidding_sequence)-1)%4]
        log.debug(f"hands: {hand}")
        log.debug(f"vul: {vul_str}")
        log.debug(f"action: {action_str}")
        log.debug(f"bidding sequence: {state.bidding_sequence}\n")

def alternate_turns(epsilon = 0.5, training = False, episode = 0):
    _states = []
    _actions = []
    _rewards = []
    _terminated = []
    state, pbn = env.reset()
    turn = (state.dealer) % 2
    terminated = False
    reward = 0
    log_game(state, pbn, -1)
    while terminated != 1:
        #print(f"bidding sequence\n{state}\nbidding sequence\n")
        #print(f"selecting action")
        log.debug("choosing action(for memory gen)")
        action = agents[turn].choose_action(state, epsilon)
        log.debug("action selected")
        #print(f"env.step(action): {action}")
        next_state, reward, terminated = env.step(action)
        _states.append(state)
        _actions.append(action)
        _rewards.append(reward)
        _terminated.append(terminated)
        log_game(next_state, pbn, action)
        turn = (turn+1)%2
        state = next_state
        if training is True:

            if len(memory) >= MIN_MEMORY_SIZE:
                exp = memory.sample()
                #print(f"type: {type(exp)}")
                #memory.log()
                agents[turn].train(exp[0])
    _rewards[-2] = -reward
    _terminated[-2] = 1
    
    for i in range (len(_states)-2):
        memory.append(Experiance(episode, _states[i], _actions[i], _rewards[i], _states[i+2], _terminated[i]))
    memory.append(Experiance(episode, _states[-2], _actions[-2], _rewards[-2], state, 1))
    memory.append(Experiance(episode, _states[-1], _actions[-1], _rewards[-1], state, 1))
    if turn%2 == 1:
        return _rewards[-1], _rewards[-2]
    else:
        return _rewards[-2], _rewards[-1]
        


def train_agents():
    epsilon_scheduler.update(episode)
    total_reward_alpha,  total_reward_beta= alternate_turns(epsilon = epsilon_scheduler.epsilon, training = True, episode = episode)
    if (episode+1) % TARGET_UPDATE == 0:
        agents[0].synchronous_networks()
        agents[1].synchronous_networks()
    alpha_rewards.append(total_reward_alpha)
    beta_rewards.append(total_reward_beta)
    log.info(f"Episode: {episode}, Total Reward, alphe: {total_reward_alpha}, beta: {total_reward_beta}")
    if (episode + 1) %50 == 0:
        plot_rewards()
def plot_rewards():
    output_dir = './pic'
    os.makedirs(output_dir, exist_ok=True)
    
    today_date = datetime.now().strftime("%Y%m%d")
    file_name = f"{today_date}_rewards_plot_episode_{episode + 1}.png"
    file_path = os.path.join(output_dir, file_name)
    
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_rewards, label='Alpha Reward', marker='o')
    plt.plot(beta_rewards, label='Beta Reward', marker='x')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(file_path)
    plt.close()
    
    log.info(f"Reward plot saved at: {file_path}")

def build_memory(min_size, initial_epsilon):
    while len(memory) < min_size:
       alternate_turns(epsilon = initial_epsilon, training = False, episode = 0)

def log_process_state():
    log.info(f"log process states: ")
    env.log()
    epsilon_scheduler.log()
    memory.log()

def save_model():
    log.info(f"save currents models: ")
    base_dir = "./models"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    today_date = datetime.now().strftime("%Y%m%d")
    existing_files = [
        f for f in os.listdir(base_dir)
        if f.startswith(today_date) and f.endswith(f".pth")
    ]
    existing_indices = []
    for f in existing_files:
        try:
            index = int(f.split("_")[1])  # date_time_<...>
            existing_indices.append(index)
        except (IndexError, ValueError):
            continue
    next_index = max(existing_indices, default=0) + 1
    try:
        new_file_path = os.path.join(base_dir, f"{today_date}_{next_index}_alpha.pth")
        agents[0].save_model(new_file_path)
        log.info(f"the first model saved as new_file_path")
        new_file_path = os.path.join(base_dir, f"{today_date}_{next_index}_beta.pth")
        agents[1].save_model(new_file_path)
        log.info(f"the second model saved as new_file_path")
    except Exception as e:
        log.error(f"main.save_model() fail to save the models: {e}")
        

def option_handler(_signum, _frame):
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
    log.info("initialize...")
    pretrained_model = torch.load(PRETRAINED_MODEL_PATH)
    #TBA
    log.info(f"load pretrained model: {PRETRAINED_MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device = {device}")
    Q_model = Duel_DDNQ(pretrained_model)
    Q_model = Q_model.to(device)
    log.info(f"initialize q network")
    optimizer = optim.Adam(Q_model.parameters(), lr=LEARNING_RATE)
    log.info(f"initialize adam optimizer, learning rate = {LEARNING_RATE}")
    agent = DuelDDQNAgent(Q_model, optimizer, DISCOUNT_FACTOR)
    agents = [copy.deepcopy(agent), copy.deepcopy(agent)]
    log.info(f"build agent list, discount factor = {DISCOUNT_FACTOR}, target q update per {TARGET_UPDATE} epochs")
    env = Env()
    log.info(f"build enviroment")
    epsilon_scheduler = EpsilonScheduler(initial_val=EPS_START, min_val=EPS_END, decay_rate=EPS_DECAY)
    log.info(f"build scheduler, initial value = {EPS_START}, minimun value = {EPS_END}, decay rate = {EPS_DECAY}")
    memory = ReplayMemory(MEMORY_SIZE)
    build_memory(MIN_MEMORY_SIZE, CREATE_MEMORY_EPSILON)
    log.info(f"build replay memory including experience {MIN_MEMORY_SIZE}, max size {MEMORY_SIZE}, epsilon used {CREATE_MEMORY_EPSILON}")
    print(f"project initialized, start training agent...")
    running = True
    run_forever = True
    steps_to_run = 0
    episode = 1
    while True:
        if running:
            # training loop
            log.info(f"episode = {episode}")
            #train_agents()
            try:
                if run_forever:
                    train_agents()
                    episode+=1
                else:
                    if steps_to_run > 0:
                        train_agents()
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
                save_model()
            elif cmd == 'q':
                log.info("process terminated")
                break
            else:
                print("invalid command.")