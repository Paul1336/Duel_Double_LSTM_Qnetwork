import unittest
from dealer import Dealer, CARD_NAMES
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class TestDealer(unittest.TestCase):
    def test_1000_new_game_pbn(self):
        num_simulations = 100
        card_to_index = {card: index for index, card in enumerate(CARD_NAMES)}
        HCP_weights = torch.tensor([4, 3, 2, 1])
        VUL_weight = torch.tensor([0, 1, 2, 3])
        card_position_count = {
            suit: {card: [0, 0, 0, 0] for card in "AKQJT98765432"} for suit in ["S", "H", "D", "C"]
        }
        vul_0_count = 0
        vul_1_count = 0
        dealer_count = [0, 0, 0, 0]
        bool_to_decimal_map = {
            (False, False): 2,  # 00
            (False, True):  3,  # 01
            (True, False):  0,  # 10
            (True, True):   1   # 11
        }
        for _ in range(num_simulations):
            deal = Dealer.new_game()
            pbn = deal.pbn[2:]
            hands = pbn.split(" ")
            vul = deal.vul
            state = deal.new_state
            self.assertIsInstance(state.features, list, "State.features sjould be a list")
            self.assertTrue(
                all(isinstance(feature, torch.Tensor) for feature in state.features),
                "State.features should include torch.Tensor elements"
            )
            self.assertTrue(
                all(feature.shape == torch.Size([66]) for feature in state.features),
                "each tensor in State.features tensor should have shape [66]"
            )
            self.assertIsInstance(state.bidding_sequence, torch.Tensor, "State.bidding_sequence should be torch.Tensor")
            self.assertEqual(state.bidding_sequence.shape[0], 0, "State.bidding_sequence should initially be empty")

            self.assertIsInstance(state.last_doubled, int, "State.last_doubled should be int")
            self.assertEqual(state.last_doubled, 0, "State.last_doubled should initially be 0")

            self.assertIsInstance(state.last_bid, int, "State.last_bid should be int")
            self.assertEqual(state.last_bid, 0, "State.last_bid should initially be 0")

            self.assertIsInstance(state.last_pass, int, "State.last_pass should be int")
            self.assertEqual(state.last_pass, 0, "State.last_pass should initially be 0")

            self.assertIsInstance(state.dealer, int, "State.dealer should be int")
            self.assertTrue(0 <= state.dealer <= 3, "State.dealer should be between 0~3")
            dealer_count[state.dealer] += 1

            vul_0_count += vul[0]
            vul_1_count += vul[1]
            self.assertEqual(len(vul), 2, "deal.vul should have exactly 2 elements")
            self.assertTrue(all(0 <= v <= 1 for v in vul), f"vul values should be 0 or 1, got {vul}")
            self.assertEqual(len(hands), 4, "hands not equal 4")
            suits_count = {
                "S": {card: 0 for card in CARD_NAMES},
                "H": {card: 0 for card in CARD_NAMES},
                "D": {card: 0 for card in CARD_NAMES},
                "C": {card: 0 for card in CARD_NAMES}
            }
            
            for i, hand in enumerate(hands):
                suits = hand.split(".")
                self.assertEqual(len(suits), 4, f"suit not equal 4: {hand}, pbn: {pbn}")
                for suit_index, suit_cards in enumerate(suits):
                    suit_name = ["S", "H", "D", "C"][suit_index]
                    for card in suit_cards:
                        self.assertIn(card, CARD_NAMES, f"invalid card: {card}, pbn: {pbn}")
                        suits_count[suit_name][card] += 1
                        card_position_count[suit_name][card][i] += 1

                        self.assertEqual(
                            state.features[i][suit_index*13+card_to_index[card]].item(), 1,
                            f"Feature mismatch: Player {i}, Suit {suit}, Card {card}, PBN {pbn}"
                        )
                for k in range (0, 4):
                    self.assertEqual(
                        (HCP_weights*state.features[i][0+13*k:4+13*k]).sum().item(), state.features[i][52+k].item(),
                        f"Sum of first four elements {(HCP_weights*state.features[i][0+13*k:4+13*k]).sum().item()} does not match tensor{52+k} = {state.features[i][52+k].item()}"
                    )
                    self.assertEqual(
                        state.features[i][0+13*k:13+13*k].sum().item(), state.features[i][57+k].item(),
                        f"lenth of suit {k} {state.features[i][0+13*k:13+13*k].sum().item()} does not match tensor{57+k} = {state.features[i][57+k].item()}"
                    )
                self.assertEqual(
                    state.features[i][52:56].sum().item(), state.features[i][56].item(),
                    f"Sum of four HCP elements {state.features[i][52:56]} does not match tensor 56 = {state.features[i][56].item()}"
                )

                if not torch.all(state.features[i][57:61] > 1):
                    self.assertEqual(
                        state.features[i][61].item(), 0,
                        f"tensor[61] should be 0 when any of tensor[57:60] <= 1, but got {state.features[i][61].item()}"
                    )
                else:
                   self.assertEqual(
                        state.features[i][61].item(), 1,
                        f"tensor[61] should be 1 when any of tensor[57:60] >=2,  but got {state.features[i][61].item()}"
                    )
                bool_tensor = torch.tensor([vul[(0+i)%2], vul[(1+i)%2]], dtype=torch.bool)
                bool_tuple = tuple(bool_tensor.tolist())
                mapped_value = bool_to_decimal_map[bool_tuple]
                self.assertEqual(
                    (VUL_weight*state.features[i][62:66]).sum().item(), mapped_value,
                    f"mapped value for vul should be {mapped_value}, but got {(VUL_weight*state.features[i][62:66]).sum().item()}"
                )
                
            for suit, cards in suits_count.items():
                for card, count in cards.items():
                    self.assertEqual(
                        count, 1,
                        f"suit {suit} card {card} should appear once instead of {count} times"
                    )

        data_list = [
            count
            for suit in card_position_count.values()
            for card_counts in suit.values()
            for count in card_counts
        ]

        data = np.array(data_list)
        
        shapiro_stat, shapiro_p = stats.shapiro(data)
        print(f"Shapiro-Wilk Test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
        
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        print(f"Kolmogorov-Smirnov Test: D={ks_stat:.4f}, p={ks_p:.4f}")
        
        anderson_result = stats.anderson(data, dist='norm')
        print(f"Anderson-Darling Test: Statistic={anderson_result.statistic:.4f}")
        for i, crit in enumerate(anderson_result.critical_values):
            print(f"  Significance Level {anderson_result.significance_level[i]}: {crit:.4f}")
        
        stats.probplot(data, dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        plt.show()

        plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label="Data Histogram")
        mu, std = np.mean(data), np.std(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, label=f"Normal PDF (mu={mu:.2f}, std={std:.2f})")
        plt.legend()
        plt.title("Histogram and Fitted Normal Distribution")
        plt.show()

        print(f"In {num_simulations} simulations:")
        print(f"vul[0] = 1 occurred {vul_0_count} times")
        print(f"vul[1] = 1 occurred {vul_1_count} times")
        print("dealer count: ", dealer_count)

        self.assertGreater(vul_0_count, 0, "vul[0] = 1 should occur at least once")
        self.assertGreater(vul_1_count, 0, "vul[1] = 1 should occur at least once")

if __name__ == "__main__":
    unittest.main()