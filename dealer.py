from dataclasses import dataclass, field
from state import State
import torch
import random
import logger

USE_CARD_DETAIL = True

@dataclass
class Deal:
    new_state: State = None
    vul: list[int] = field(default_factory=lambda: [0, 0])
    pbn: str = None

def shuffle(deck:list):
    #Fisher-Yates shuffle
    for i in range(len(deck) - 1, 0, -1):
        j = random.randint(0, i)
        deck[i], deck[j] = deck[j], deck[i]
    return deck

CARD_NAMES = "AKQJT98765432"
CARD_HCP = [4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
class Dealer():
    @classmethod
    def new_game(cls)->Deal:
        log = logger.get_logger(__name__)
        deck = [i for i in range (0, 52)] #spade AKQJ...
        deck = shuffle(deck)
        _deal = Deal()
        _deal.vul[0] = random.randint(0, 1)
        _deal.vul[1] = random.randint(0, 1)
        _deal.pbn = "N:"
        _extract_features = [torch.zeros(10, dtype=torch.float32) for _ in range(4)]
        _cards = [torch.zeros(52, dtype=torch.float32) for _ in range(4)]
        for i in range(0, 4):
            hand = sorted(deck[13*i:13*(i+1)])
            suit = 0
            suit_len = 0
            bal = True
            for card in hand:
                _cards[i][card] = 1
                while(card//13 > suit):
                    _deal.pbn += "."
                    _extract_features[i][5+suit] = suit_len
                    suit += 1
                    if(suit_len < 2):
                        bal = False
                    suit_len = 0
                _deal.pbn += CARD_NAMES[card%13]
                _extract_features[i][suit] += CARD_HCP[card%13]
                suit_len += 1
            _extract_features[i][5+suit] = suit_len
            if(suit_len < 2):
                bal = False
            if bal is True:
                _extract_features[i][9] = 1
            for j in range(0, 4):
                _extract_features[i][4] += _extract_features[i][j]
            while suit < 3:
                suit += 1
                _deal.pbn += "."
            if i<3:
                _deal.pbn+=" "
        _state = State()

        _state.bidding_sequence = torch.tensor([])
        _vulnerable = [torch.zeros(4, dtype=torch.float32) for _ in range(4)]

        if(_deal.vul[0] == 1):
            if(_deal.vul[1] == 1):
                _vulnerable[0][1] = 1
                _vulnerable[1][1] = 1
                _vulnerable[2][1] = 1
                _vulnerable[3][1] = 1
            else:
                _vulnerable[0][0] = 1
                _vulnerable[1][3] = 1
                _vulnerable[2][0] = 1
                _vulnerable[3][3] = 1
        else:
            if(_deal.vul[1] == 1):
                _vulnerable[0][3] = 1
                _vulnerable[1][0] = 1
                _vulnerable[2][3] = 1
                _vulnerable[3][0] = 1
            else:
                _vulnerable[0][2] = 1
                _vulnerable[1][2] = 1
                _vulnerable[2][2] = 1
                _vulnerable[3][2] = 1
        if USE_CARD_DETAIL is True:
            _features = [torch.cat((cards, features)) for cards, features in zip(_cards, _extract_features)]
        _features = [torch.cat((features, vulnerable)) for features, vulnerable in zip(_features, _vulnerable)]
        _state.features = _features
        _state.dealer = random.randint(0, 3)
        _state.agent_team = random.randint(0, 1)
        _state.last_bid = 0
        _state.last_doubled = 0
        _state.last_pass = 0
        _deal.new_state = _state
        log.debug("Creacted a deal:")
        _deal.new_state.log()
        log.debug(f"vul: {_deal.vul}")
        log.debug(f"pbn: {_deal.pbn}")
        return _deal