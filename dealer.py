from dataclasses import dataclass
import torch
import random

@dataclass
class State:
    features: torch.tensor = None
    bidding_sequence: torch.tensor = None

@dataclass
class Deal:
    new_state: State = None
    vul: list[int] = None
    pbn: str = None

def shuffle(deck:list):
    #Implementation Fisher-Yates shuffle
    for i in range(len(deck) - 1, 0, -1):
        j = random.randint(0, i)
        deck[i], deck[j] = deck[j], deck[i]
    return deck

CARD_NAMES = "AKQJT98765432"
class Dealer():
    @classmethod
    def new_game()->Deal:
        deck = [i for i in range (0, 52)] #spade AKQJ...
        deck = shuffle(deck)
        newDeal = Deal()
        newDeal.vul[0] = random.randint(0, 1)
        newDeal.vul[1] = random.randint(0, 1)
        newDeal.pbn = "N:"
        for i in range(0, 4):
            hand = deck[13*i:13*(i+1)].sort
            suit = 0
            for card in range(0, 13):
                while(card//13 > suit):
                    newDeal.pbn += "."
                    suit += 1
                newDeal.pbn += CARD_NAMES[card%13]
            if i < 3:
                newDeal.pbn += " "
        init_state = State()
        init_state.bidding_sequence = list()
        newDeal.new_state = init_state
        return newDeal
        #tba