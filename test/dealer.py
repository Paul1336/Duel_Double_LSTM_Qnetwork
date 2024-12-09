from dealer import *
    
def _test():
    deal = Dealer.new_game()
    print(deal)
    State.print_state(deal.new_state)
    print("vul:", deal.vul)
    print("pbn:", deal.pbn)

if __name__ == "__main__":
    _test()