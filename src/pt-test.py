import torch
import time

def run_sum(size: int) -> None:
    ###CPU
    start_time = time.time()
    a = torch.ones(size, size)
    for _ in range(1000):
        a += a
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print('CPU time = ',elapsed_time)

    ###GPU
    start_time = time.time()
    b = torch.ones(size, size).cuda()
    for _ in range(1000):
        b += b
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print('GPU time = ',elapsed_time)

for size in 128, 256, 512, 1024, 2048, 4096:

    print(f'size = {size} x {size}')
    run_sum(size)
