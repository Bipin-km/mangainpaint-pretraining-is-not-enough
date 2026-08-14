"""DDP setup/teardown helpers, shared by every training run."""
import os
import random
import numpy as np
import torch
import torch.distributed as dist


def setup_ddp(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank): return rank == 0
def unwrap(m): return m.module if hasattr(m, 'module') else m


def call_g(G, inp, batch, device):
    """Calls G(inp), or G(inp, category) for generators that opt in via a
    `wants_category = True` class attribute (currently only
    `model_lcg.MangaFillNetLCG`) -- keeps every other generator's plain
    `forward(x)` signature untouched rather than threading an unused
    `cat` arg through every model_fn in the codebase."""
    if getattr(unwrap(G), "wants_category", False):
        return G(inp, batch["is_balloon"].to(device, non_blocking=True))
    return G(inp)


def reduce_mean(t):
    if not dist.is_initialized():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / dist.get_world_size()


def reduce_mean_of(values, device):
    """Global mean of a per-rank list of floats (NaNs = invalid sample,
    excluded), pooled across all ranks by summing (sum, count) rather than
    reducing a pre-averaged per-rank mean.

    This matters once metrics are stratified (e.g. by ink-density bucket):
    a per-rank bucket can easily be empty or all-NaN while another rank's
    isn't, and `reduce_mean` requires every rank to call the collective the
    same number of times -- a naive "skip reduce_mean if this rank's mean is
    NaN" would desync ranks and hang NCCL. Summing unconditionally sidesteps
    that: every rank always calls this once per stat, valid or not.
    """
    valid = [v for v in values if not np.isnan(v)]
    t = torch.tensor([float(sum(valid)), float(len(valid))], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total_sum, total_n = t[0].item(), t[1].item()
    return (total_sum / total_n if total_n > 0 else float('nan')), int(total_n)


def reduce_count(n_local, device):
    t = torch.tensor(float(n_local), device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def broadcast_state(D, src=0):
    """Sync D parameters across all ranks after refresh."""
    if not dist.is_initialized():
        return
    for p in unwrap(D).parameters():
        dist.broadcast(p.data, src=src)


def seed_everything(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
