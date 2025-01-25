import ctypes
from ctypes import c_char_p, c_int, POINTER, Structure
import platform
from dataclasses import dataclass
from state import State
import logger


DLL_PATH = "./DoubleDummySolver.dll"
SO_PATH = "./linux_dds.so"


@dataclass
class ddResponse(ctypes.Structure):
    _fields_ = [("NS_imp_loss", ctypes.c_int),
                ("error_type_calc", ctypes.c_int),
                ("error_type_par", ctypes.c_int),]

class RewardCalculator:
    pbn_str: str
    vul: list[int]
    dll: ctypes.CDLL

    def __init__ (self, _deal: str, _vul:list[int]):
        # _deal: pbn, vul: vul[]
        self.pbn_str = _deal
        self.vul = _vul
        #deal = ctypes.c_char_p(b"N:A2.AKQJT98765..2 T987.3.AT987.876 KQJ.2.KQJ.AKQJT9 6543.4.65432.543")# NESW, SHDC, from deal
        #vul = (ctypes.c_int * 2)(1, 0)# from deal
        try:
            if platform.system() == "Windows":
                self.dll = ctypes.CDLL(DLL_PATH)
            else:
                #print(SO_PATH)
                self.dll = ctypes.CDLL(SO_PATH)
        except Exception as e:
            raise RuntimeError(f"RewardCalculator.init() fail to load the .dll/.so: {e}") from e
        self.dll.ddAnalize.restype = ddResponse
        self.dll.ddAnalize.argtypes = [
            c_char_p,            # deal (PBN string)
            POINTER(c_int),      # vul (array of 2 integers)
            c_int,               # AP_hand
            c_int,               # suit
            c_int,               # level
            c_int,               # doubled
            c_int,               # dealer
        ]

    def imp_diff (self, state:State, sequence_end = True) -> tuple[int, int]:
        if sequence_end == False:
            pass
        else:
        # calculete the imp loss of the last bid in the bidding sequence
            try:
                log = logger.get_logger(__name__)
                AP_hand = False
                suit = 0 # C, D, H, S
                level = 0 #0~6
                doubled = 0
                dealer = -1 #NESW
                if len(state.bidding_sequence) <= 4 and state.last_bid==0:
                    AP_hand = True
                else:
                    #view = (len(state.bidding_sequence) + state.dealer +1) % 2

                    suit = (state.bidding_sequence[-state.last_bid]-3) % 5
                    level = (state.bidding_sequence[-state.last_bid]-3) // 5
                    if state.last_bid > state.last_doubled and state.last_doubled != 0:
                        doubled = state.bidding_sequence[-state.last_doubled]
                    else:
                        doubled = 0
                    for index, value in enumerate(state.bidding_sequence):
                        #print("index: ", index, "value: ", value)
                        if index%2 == (len(state.bidding_sequence)-state.last_bid)%2:
                            #print("alpha, value: ", value, "suit: ", suit)
                            if (value-3)%5 == suit:
                                #print("beta")
                                dealer = (index+state.dealer)%4
                                break
                #print("*self.vul", *self.vul)
                #print("suit", suit)
                #print("level", level)
                #print("doubled", doubled)
                #print("dealer", dealer)
                #print("pbn_str", self.pbn_str)
                res = self.dll.ddAnalize(ctypes.c_char_p(self.pbn_str.encode('utf-8')), 
                                        (ctypes.c_int * 2)(*self.vul), ctypes.c_int(AP_hand), ctypes.c_int(suit), 
                                        ctypes.c_int(level), ctypes.c_int(doubled),ctypes.c_int(dealer))
                #res = self.dll.ddAnalize(ctypes.c_char_p(self.pbn_str.encode('utf-8')), (ctypes.c_int * 2)(*self.vul), ctypes.c_int(suit), ctypes.c_int(level), ctypes.c_int(doubled), ctypes.c_int(dealer))
                if res.error_type_calc != 1:
                    raise RuntimeError(f"CalcDDtablePBN() error type :{res.error_type_calc}")
                if res.error_type_par != 1:
                    raise RuntimeError(f"Par() error type :{res.error_type_par}")
                
                log.debug(f"calculating imp loss, given state: ")
                log.debug(f"pbn: {self.pbn_str}")
                log.debug(f"vul: {self.vul}")
                log.debug(f"suit = {suit}, level = {level}, doubled = {doubled}, dealer = {dealer}")
                log.debug(f"imp loss = NS: {res.NS_imp_loss}, EW: {-res.NS_imp_loss}")
                return res.NS_imp_loss, -res.NS_imp_loss

            except Exception as e:
                raise RuntimeError(f"An error occur in RewardCalculator.imp_diff(): {e}") from e
    def __deepcopy__(self, memo):
        # Exclude the DLL object from being copied
        copied = RewardCalculator(self.pbn_str, self.vul)
        copied.dll = self.dll  # Reuse the same DLL instance
        return copied