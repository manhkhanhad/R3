# test_nccl.py
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")

x = torch.tensor([rank], dtype=torch.float32, device="cuda")

print(f"rank {rank}: before {x}", flush=True)

dist.all_reduce(x)

print(f"rank {rank}: after {x}", flush=True)

dist.destroy_process_group()