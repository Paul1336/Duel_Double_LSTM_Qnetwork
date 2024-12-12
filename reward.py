import ctypes
import platform
from dataclasses import dataclass
from state import State
import logger


DLL_PATH = "./DoubleDummySolver.dll"
SO_PATH = "./linux_dds.so"


@dataclass
class ddResponse(ctypes.Structure):
    _fields_ = [("imp_loss", ctypes.c_int),
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
                self.dll = ctypes.CDLL(SO_PATH)
        except Exception as e:
            raise RuntimeError(f"RewardCalculator.init() fail to load the .dll/.so: {e}") from e
        self.dll.ddAnalize.restype = ddResponse

    def imp_diff (self, state:State) -> float:
        # calculete the imp loss of the last bid in the bidding sequence
        try:
            log = logger.get_logger(__name__)
            suit = -1 # C, D, H, S
            level = -1 #0~6
            doubled = -1
            dealer = -1
            view = (len(state.bidding_sequence) + state.dealer -1) % 4
            if len(state.bidding_sequence) > 4:
                suit = (state.bidding_sequence[-state.last_bid]-3) % 5
                level = (state.bidding_sequence[-state.last_bid]-3) // 5
                if(state.last_bid > state.last_doubled):
                    doubled = state.bidding_sequence[-state.last_doubled]
                for index, value in enumerate(state.bidding_sequence):
                    if index%2 == len(state.bidding_sequence -1 )%2:
                        if (value-3)%5 == suit:
                            dealer = (index+dealer)%4
                            break
            #print("*self.vul", *self.vul)
            #print("suit", suit)
            #print("level", level)
            #print("doubled", doubled)
            #print("dealer", dealer)
            #print("pbn_str", self.pbn_str)
            res = self.dll.ddAnalize(ctypes.c_char_p(self.pbn_str.encode('utf-8')), 
                                     (ctypes.c_int * 2)(*self.vul), ctypes.c_int(suit), 
                                     ctypes.c_int(level), ctypes.c_int(doubled), 
                                     ctypes.c_int(dealer), ctypes.c_int(view))
            #res = self.dll.ddAnalize(ctypes.c_char_p(self.pbn_str.encode('utf-8')), (ctypes.c_int * 2)(*self.vul), ctypes.c_int(suit), ctypes.c_int(level), ctypes.c_int(doubled), ctypes.c_int(dealer))
            if res.error_type_calc != 0:
                raise RuntimeError(f"CalcDDtablePBN() error type :{res.error_type_calc}")
            if res.error_type_par != 0:
                raise RuntimeError(f"Par() error type :{res.error_type_par}")
            log.debug(f"calculating imp loss, given state: ")
            log.debug(f"pbn: {self.pbn_str}")
            log.debug(f"vul: {self.vul}")
            log.debug(f"suit = {suit}, level = {level}, doubled = {doubled}, dealer = {dealer}, view = {view}")
            log.debug(f"imp loss = {res.imp_loss}")
            return res.imp_loss

        except Exception as e:
            raise RuntimeError(f"An error occur in RewardCalculator.imp_diff(): {e}") from e