import os
import torch
import torch.distributed as dist

def setup_dist():
    # torchrun 自动注入环境变量 RANK LOCAL_RANK WORLD_SIZE
    assert "LOCAL_RANK" in os.environ, "LOCAL_RANK not set"
    assert "RANK" in os.environ, "RANK not set"

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")

    print(f"[Rank {rank}] init OK. Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")

def main():
    setup_dist()

    rank = dist.get_rank()
    world = dist.get_world_size()

    tensor = torch.tensor([rank], device=torch.cuda.current_device())
    gathered = [torch.zeros_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor)

    print(f"[Rank {rank}] allgather -> {[int(x.item()) for x in gathered]}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
