import unittest
from env import Env, Experiance
from state import State
import torch
import copy

#random, reward
class TestEnv(unittest.TestCase):

    def setUp(self):
        self.env = Env()

    def test_reset(self):
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
        _, _, terminate = self.env.step(0)
        self.assertEqual(terminate, 1, "Shoud return terminate = 1 after 3 passes")
        self.env.reset()
        self.env.step(0)
        self.env.step(0)
        self.env.step(0)
        _, _, terminate = self.env.step(0)
        self.assertEqual(terminate, 1, "Shoud return terminate = 1 after 3 passes")
        
    def test_step(self):
        self.env.reset()
        state_before = self.env.current_state
        action = Env.random_action(state_before)
        next_state, reward, terminated = self.env.step(action)
        self.assertIsInstance(next_state, State, "Step should return the next state.")
        self.assertIsInstance(reward, int, "Reward should be a int.")
        self.assertIsInstance(terminated, int, "Terminated should be an integer.")
        self.assertGreaterEqual(terminated, 0, "Terminated should be 0 or 1.")
        self.assertLessEqual(terminated, 1, "Terminated should be 0 or 1.")

   
def play_multiple_games(env: Env, num_games: int):

    results = []

    for game in range(num_games):
        print(f"Starting game {game + 1}...")
        env.reset()
        game_result = {
            "actions": [],
            "reward": 0
        }
        terminated = 0
        print("initial state: ", env.current_state)
        while not terminated:
            action = Env.random_action(env.current_state)
            game_result["actions"].append(action)
            print(f"action:{action}, last_doubled={env.current_state.last_doubled}, last_bid={env.current_state.last_bid}, last_pass={env.current_state.last_pass}")
            next_state, reward, terminated = env.step(action)
            game_result["reward"] += reward
            #print(f"current state: sequence {env.current_state.bidding_sequence}, last_bid {env.current_state.last_bid}, last_pass {env.current_state.last_pass}, last_doubled {env.current_state.last_doubled}")
        print("terminate state: ", env.current_state)
        results.append(game_result)
        print(f"Game {game + 1} finished with reward: {game_result['reward']}, actions: {game_result['actions']}")

    return results


if __name__ == "__main__":
    
    env = Env()
    num_games = 1
    results = play_multiple_games(env, num_games)
    for i, result in enumerate(results):
        print(f"Game {i + 1}:")
        print(f"  Actions: {result['actions']}")
        print(f"  Total Reward: {result['reward']}")
    unittest.main()