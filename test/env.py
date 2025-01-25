import unittest
from env import Env, Experiance
from state import State
import torch
import copy

def test_reward(env: Env):
    print("reward test")
    
    _env = copy.deepcopy(env)
    zero_cnt = 0
    i = 1
    reward = -1000
    if len(_env.current_state.bidding_sequence) == 0:
        zero_cnt = -1
    else:
        while i<=len(_env.current_state.bidding_sequence) and _env.current_state.bidding_sequence[-i] == 0:
            zero_cnt+=1
            i+=1
    print(f"original bidding sequence: {_env.current_state.bidding_sequence}")
    for i in range(0, 3-zero_cnt):
        _, reward, _ = _env.step(0)
    if zero_cnt == 3:
        if len(_env.current_state.bidding_sequence)%2 == 1:
            reward, _ = _env.reward_calculater.imp_diff(_env.current_state)
        else: 
            _, reward = _env.reward_calculater.imp_diff(_env.current_state)
    print(f"bidding sequence for test: {_env.current_state.bidding_sequence}")
    return reward

#random, reward
class TestEnv(unittest.TestCase):

    def setUp(self):
        self.env = Env()

    def test_reset(self):
        print("in test_reset: ")
        initial_stste = copy.deepcopy(self.env.current_state)
        self.env.reset()
        reset_stste = copy.deepcopy(self.env.current_state)
        vars_to_check = [reset_stste.last_bid, reset_stste.last_doubled, reset_stste.last_pass, initial_stste.last_bid, initial_stste.last_doubled, initial_stste.last_pass]
        self.assertTrue(
            all(v == 0 for v in vars_to_check),
            f"Not all variables are 0: {vars_to_check}"
        )
        self.assertEqual(len(reset_stste.bidding_sequence), 0, "Bidding sequence should be empty after reset")

    def test_record_bidding(self):
        print("in test_record_bidding: ")
        print("======================================================================================")
        print("initial state 1: ", self.env.current_state)
        print("======================================================================================")
        state = self.env.current_state
        self.env.step(0)
        self.assertEqual(state.last_bid, 0, "Last bid should be 0 after the first pass.")
        self.assertEqual(state.last_pass, 1, "Last pass should be updated to 1 after a pass.")
        self.assertEqual(state.last_doubled, 0, "Last doubled should be 0 after the first passd.")
        self.env.step(4)
        self.assertEqual(state.last_bid, 1, "Last bid should be updated to 1 after a normal bid.")
        self.assertEqual(state.last_pass, 2, "Last pass should be updated to 2 after a normal bid.")
        self.assertEqual(state.last_doubled, 0, "Last doubled should be 0 before any double.")
        self.env.step(0)
        self.assertEqual(state.last_bid, 2, "Last bid should be updated to 2 after a pass.")
        self.assertEqual(state.last_pass, 1, "Last pass should be updated to 1 after a pass.")
        self.assertEqual(state.last_doubled, 0, "Last doubled should be 0 before any double.")
        self.env.step(0)
        self.assertEqual(state.last_bid, 3, "Last bid should be updated to 3 after two pass.")
        self.assertEqual(state.last_pass, 1, "Last pass should be updated to 1 after a pass.")
        self.assertEqual(state.last_doubled, 0, "Last doubled should be 0 before any double.")
        self.env.step(1)
        self.assertEqual(state.last_bid, 4, "Last bid should be updated to 4 after two pass and a double.")
        self.assertEqual(state.last_pass, 2, "Last pass should be updated to 2 after a double.")
        self.assertEqual(state.last_doubled, 1, "Last doubled should be updated to 1 after a double.")
        self.env.step(2)
        self.assertEqual(state.last_bid, 5, "Last bid should be updated to 5 after two pass, a double, and a redouble.")
        self.assertEqual(state.last_pass, 3, "Last pass should be updated to 3 after a double and a redouble.")
        self.assertEqual(state.last_doubled, 1, "Last doubled should be updated to 1 after a redouble.")
        self.env.step(7)
        self.assertEqual(state.last_bid, 1, "Last bid should be updated to 1 after a normal bid.")
        self.assertEqual(state.last_pass, 4, "Last pass should be updated to 4 after a double, a redouble, and a normal bid.")
        self.assertEqual(state.last_doubled, 2, "Last redouble should be updated to 2 after a normal bid.")
        self.env.step(0)
        self.assertEqual(state.last_bid, 2, "Last bid should be updated to 2 a pass.")
        self.assertEqual(state.last_pass, 1, "Last pass should be updated to 1 after a pass.")
        self.assertEqual(state.last_doubled, 3, "Last redouble should be updated to 2 after a normal bid and a double.")
        self.env.step(0)
        print("======================================================================================")
        print("before terminate 1")
        print("======================================================================================")
        _, _, terminate = self.env.step(0)
        print("======================================================================================")
        print("terminate 1: ", self.env.current_state)
        print("======================================================================================")
        self.assertEqual(terminate, 1, "Shoud return terminate = 1 after 3 passes")
        self.env.reset()
        print("======================================================================================")
        print("initial state 2: ", self.env.current_state)
        print("======================================================================================")
        self.env.step(0)
        self.env.step(0)
        self.env.step(0)
        print("======================================================================================")
        print("before terminate 2")
        print("======================================================================================")
        _, _, terminate = self.env.step(0)
        print("======================================================================================")
        print("terminate 2: ", self.env.current_state)
        print("======================================================================================")
        self.assertEqual(terminate, 1, "Shoud return terminate = 1 after 3 passes")
        
    def test_step(self):
        print("in test_step: ")
        self.env.reset()
        state_before = self.env.current_state
        action = Env.random_action(state_before)
        next_state, reward, terminated = self.env.step(action)
        self.assertIsInstance(next_state, State, "Step should return the next state.")
        self.assertIsInstance(reward, int, "Reward should be a int.")
        self.assertIsInstance(terminated, int, "Terminated should be an integer.")
        self.assertGreaterEqual(terminated, 0, "Terminated should be 0 or 1.")
        self.assertLessEqual(terminated, 1, "Terminated should be 0 or 1.")

   
    def test_play_multiple_games(self):
        print("in test_play_multiple_games: ")
        num_games = 100
        results = []
        env = Env()
        for game in range(num_games):
            print("======================================================================================")
            print(f"Starting game {game + 1}...")
            env.reset()
            game_result = {
                "actions": [],
                "reward": 0
            }
            terminated = 0
            print("======================================================================================")
            print("initial state: ", env.current_state)
            print("======================================================================================")
            print(f"AP reward = {test_reward(env)}")
            print("======================================================================================")
            while not terminated:
                action = Env.random_action(env.current_state)
                game_result["actions"].append(action)
                next_state, reward, terminated = env.step(action)
                print("======================================================================================")
                print(f"action:{action}\nlast_doubled={env.current_state.last_doubled}, last_bid={env.current_state.last_bid}, last_pass={env.current_state.last_pass}")
                print(f"reward if AP = {test_reward(env)}")
                print("======================================================================================")
                game_result["reward"] += reward
                #print(f"current state: sequence {env.current_state.bidding_sequence}, last_bid {env.current_state.last_bid}, last_pass {env.current_state.last_pass}, last_doubled {env.current_state.last_doubled}")
            print("======================================================================================")
            print("terminate state: ", env.current_state)
            print("======================================================================================")
            print(f"reward = {test_reward(env)}")
            print("======================================================================================")
            results.append(game_result)
            print(f"Game {game + 1} finished with reward: {game_result['reward']}, actions: {game_result['actions']}")
            zero_cnt = 0
            double_cnt = 0
            max_bid = -1
            for i in range(0, len(env.current_state.bidding_sequence)):
                if env.current_state.bidding_sequence[i] == 0:
                    zero_cnt += 1
                else:
                    zero_cnt = 0
                if env.current_state.bidding_sequence[i] == 1:
                    self.assertEqual(double_cnt, 0, "shouldn't have double over double")
                    double_cnt = 1
                if env.current_state.bidding_sequence[i] == 2:
                    self.assertEqual(double_cnt, 1, "should have double before redouble")
                    if i > 1:
                        self.assertNotEqual(env.current_state.bidding_sequence[i-2], 1, "shouldn't redouble against partner")
                    double_cnt = 2
                        
                if env.current_state.bidding_sequence[i] > 2:
                    double_cnt = 0
                    self.assertGreater(env.current_state.bidding_sequence[i], max_bid, "insufficient bid")
                    max_bid = env.current_state.bidding_sequence[i]
                self.assertLess(zero_cnt, 4, "continous zero shouldn't exceed 3 time")


if __name__ == "__main__":
    unittest.main()