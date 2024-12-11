from dealer import *
    
def _test():
    deal = Dealer.new_game()
    with open("./test.log", 'a') as f:
        deal.new_state.log(f)
    print("vul:", deal.vul)
    print("pbn:", deal.pbn)

if __name__ == "__main__":
    _test()