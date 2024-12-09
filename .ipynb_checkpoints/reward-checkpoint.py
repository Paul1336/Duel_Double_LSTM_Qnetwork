import ctypes
from dataclasses import dataclass
from state import State

DLL_PATH = "./DoubleDummySolver.dll"
SO_PATH = "./linux_dds.so"


@dataclass
class ddResponse(ctypes.Structure):
    _fields_ = [("imp_loss", ctypes.c_int),
                ("error_type", ctypes.c_int),]
    # levels: [vul][suit][player]

class RewardCalculator:
    pbn_str: str
    vul: list[int]
    dll: ctypes.CDLL = None

    def __init__ (self, _deal: str, _vul:list[int]): #tba deal type
        self.pbn_str = _deal
        self.vul = _vul
        #deal = ctypes.c_char_p(b"N:A2.AKQJT98765..2 T987.3.AT987.876 KQJ.2.KQJ.AKQJT9 6543.4.65432.543")# NESW, SHDC, from deal
        #vul = (ctypes.c_int * 2)(1, 0)# from deal
        self.dll = ctypes.CDLL(SO_PATH)
        self.dll.ddAnalize.restype = ddResponse

    def imp_diff (self, state:State) -> float:
        suit = -1
        level = 0
        doubled = 0
        dealer = -1
        if len(state.bidding_sequence) > 4:
            suit = (state.bidding_sequence[state.last_bid]-3)%5
            level = (state.bidding_sequence[state.last_bid]-3)/5
            if(state.last_bid > state.last_doubled):
                doubled = state.bidding_sequence[state.last_doubled]
            for index, value in enumerate(state.bidding_sequence):
                if index%2 == len(state.bidding_sequence)%2:
                    if (value-3)%5 == suit:
                        dealer = index%4
        res = self.dll.ddAnalize(ctypes.c_char_p(self.pbn_str.encode('utf-8')), (ctypes.c_int * 2)(*self.vul), ctypes.c_int(suit), ctypes.c_int(level), ctypes.c_int(doubled), ctypes.c_int(dealer))
        if res.error_type[0] != 0 or self.imp_chart.error_type[1] != 0:
            print("err")
            return 0
        return res.imp_loss
        # tba, record dealer of suit and return loss


from dealer import Dealer
def _test():
    deal = Dealer.new_game()
    calculator = RewardCalculator(deal.pbn, deal.vul)
    deal.new_state.bidding_sequence = [0, 0, 6, 1, 8, 0, 9, 0, 21]
    deal.new_state.print_state
    print(calculator.imp_diff(deal.new_state))
    deal.new_state.bidding_sequence = [0, 0, 6, 1, 8, 0, 9, 0, 21, 0, 0, 0]
    deal.new_state.print_state
    print(calculator.imp_diff(deal.new_state))

if __name__ == "__main__":
    _test()