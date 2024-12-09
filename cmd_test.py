import signal
import time
import os

pid = os.getpid()
print("目前程式的 PID 是:", pid)
# 狀態變數
running = True  # 控制迴圈是否繼續執行

def snapshot_handler(signum, frame):
    # 當收到 SIGUSR1 時執行的動作，例如輸出當前程式的內部狀態
    print("=== SNAPSHOT ===")
    # 在這裡列印出你想要的資訊，例如目前迴圈次數、重要參數等
    # ...

def pause_resume_handler(signum, frame):
    # 收到 SIGUSR2 時，切換 running 狀態
    global running
    running = not running
    if running:
        print("Loop resumed.")
    else:
        print("Loop paused.")

signal.signal(signal.SIGUSR1, snapshot_handler)
signal.signal(signal.SIGUSR2, pause_resume_handler)

i = 0
while True:
    if running:
        i += 1
        # 模擬運算
        time.sleep(0.5)
    else:
        # 暫停時可選擇做點別的事或單純空轉
        time.sleep(0.1)
