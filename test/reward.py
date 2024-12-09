from dealer import *
from reward import *

def _test():
    deal = Dealer.new_game()
    deal.new_state.print_state()
    print(deal.pbn)
    calculator = RewardCalculator(deal.pbn, deal.vul)
    deal.new_state.bidding_sequence = [0, 0, 6, 1, 8, 0, 9, 0, 21]
    deal.new_state.print_state
    print(calculator.imp_diff(deal.new_state))
    deal.new_state.bidding_sequence = [0, 0, 6, 1, 8, 0, 9, 0, 21, 0, 0, 0]
    deal.new_state.print_state
    print(calculator.imp_diff(deal.new_state))

if __name__ == "__main__":
    _test()