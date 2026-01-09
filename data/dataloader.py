import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root="edu_fineweb10B", master_process=True):
        self.B = B
        self.T = T
        self.rank = process_rank
        self.world_size = num_processes
        self.master_process = master_process

        assert split in {"train", "val"}

        shards = sorted(
            os.path.join(data_root, f)
            for f in os.listdir(data_root)
            if split in f
        )
        assert len(shards) > 0, f"no shards found for split {split}"
        self.shards = shards
        self.num_shards = len(shards)

        if master_process:
            logger.info(f"found {self.num_shards} shards for split {split}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.max_batches_per_shard = (len(self.tokens)-1) // (self.B*self.T*self.world_size)
        self.batch_idx = 0

    def next_batch(self):
        B, T = self.B, self.T
        ws = self.world_size
        idx = self.batch_idx * ws + self.rank
        start = idx * B * T
        end = start + B * T + 1

        buf = self.tokens[start:end]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Advance GLOBAL batch counter
        self.batch_idx += 1

        # Shard transition happens at SAME time on all ranks
        if self.batch_idx >= self.max_batches_per_shard:
            self.current_shard = (self.current_shard + 1)%self.num_shards
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.max_batches_per_shard = (len(self.tokens)-1)//(B*T*ws)
            self.batch_idx = 0

        return x, y