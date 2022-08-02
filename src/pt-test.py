import torch
import time

def run_sum(size: int) -> None:
    ###CPU
    start_time = time.time()
    a = torch.ones(size, size)
    for _ in range(1000):
        a += a
    torch.cuda.synchronize()
    cpu_elapsed_time = time.time() - start_time

    print(f'CPU time = {cpu_elapsed_time: .3f}')

    ###GPU
    start_time = time.time()
    b = torch.ones(size, size).cuda()
    for _ in range(1000):
        b += b
    torch.cuda.synchronize()
    gpu_elapsed_time = time.time() - start_time

    print(f'GPU time = {gpu_elapsed_time: .3f}')
    print(f'Speedup = {cpu_elapsed_time/gpu_elapsed_time: .3f}')
    
for size in 128, 256, 512, 1024, 2048, 4096, 8192:
    print(f'size = {size} x {size}')
    run_sum(size)
