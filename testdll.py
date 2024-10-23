import ctypes

class ddResponse(ctypes.Structure):
    _fields_ = [("imp_loss", ctypes.c_int * 7 * 5 * 4 * 3),
                ("error_type", ctypes.c_int * 2),]
    # levels: [vul][suit][player]


dll_path = "./DoubleDummySolver.dll"
my_str = "N:A2.AKQJT98765..2 T987.3.AT987.876 KQJ.2.KQJ.AKQJT9 6543.4.65432.543"
my_str = "N:AKQJT98765432... .AKQJT98765432.. ..KQJT98765432.2 ..A.AKQJT9876543"
my_list = [1, 0]
deal = ctypes.c_char_p(my_str.encode('utf-8'))# NESW, SHDC
vul = (ctypes.c_int * 2)(*my_list)
dll = ctypes.CDLL(dll_path)
dll.ddAnalize.restype = ddResponse
result = dll.ddAnalize(deal, vul)

print(f"error_type: {result.error_type[0]}, {result.error_type[1]}")
imp_loss = [[[[result.imp_loss[v][p][s][m] for m in range(7)] for s in range(5)] for p in range(4)] for v in range(3)]

print(f"imp_loss: {imp_loss}")
print(f"undoubled: {imp_loss[0][0]}")
