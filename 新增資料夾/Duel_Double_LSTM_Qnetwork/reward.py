from typing import List
import ctypes
from dataclasses import dataclass

DLL_PATH = "./DoubleDummySolver.dll"

@dataclass
class ddResponse(ctypes.Structure):
    _fields_ = [("imp_loss", ctypes.c_int * 7 * 5 * 4 * 3),
                ("error_type", ctypes.c_int * 2),]
    # levels: [vul][suit][player]

class RewardCalculator:
    dealer: List[List[int]] # NS/EW, suit
    imp_chart: ddResponse

    def __init__ (self, deal: str, vul:list[int]): #tba deal type
        self.dealer = [[2 for j in range(2)] for i in range(5)]
        #deal = ctypes.c_char_p(b"N:A2.AKQJT98765..2 T987.3.AT987.876 KQJ.2.KQJ.AKQJT9 6543.4.65432.543")# NESW, SHDC, from deal
        #vul = (ctypes.c_int * 2)(1, 0)# from deal
        dll = ctypes.CDLL(DLL_PATH)
        dll.ddAnalize.restype = ddResponse
        self.imp_chart = dll.ddAnalize(ctypes.c_char_p(deal.encode('utf-8')), (ctypes.c_int * 2)(*vul))
        if self.imp_chart.error_type[0] != 0 or self.imp_chart.error_type[1] != 0:
            print("err")
            

    def imp_diff (self, dealer, bid) -> float:
        return 0.5 
        # tba, record dealer of suit and return loss