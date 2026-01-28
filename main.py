import os
import sys
from pathlib import Path
import torch
import tiktoken
from models import GPT, GPTConfig
from data.dataloader import DataLoaderLite
from torch.nn.parallel import DistributedDataParallel as DDP
from training.trainer import Trainer, setup_ddp
from logger import setup_logging

os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
log_dir = PROJECT_ROOT / "log"
logger = setup_logging(log_dir=log_dir)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_ddp()

# Config
enc = tiktoken.get_encoding("gpt2")
total_batch_size = 50257
B = 64
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
data_dir = PROJECT_ROOT / "edu_fineweb10B"

if master_process:
    logger.info(f"total desired batch size: {total_batch_size}")
    logger.info(f"Calculated gradient accumulation steps: {grad_accum_steps}")

# Data loaders
train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, 
    num_processes=ddp_world_size, split="train",
    master_process=master_process,
    data_root=data_dir
)
val_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, 
    num_processes=ddp_world_size, split="val",
    master_process=master_process,
    data_root=data_dir
)

torch.set_float32_matmul_precision('high')

# Create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

use_compile = False  # interferes with HellaSwag eval and Generation
if use_compile:
    model = torch.compile(model)

if ddp:
    print(f"Rank {ddp_rank} reached DDP")
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# Optimizer
device_type = "cuda" if device.startswith("cuda") else "cpu"
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, 
    learning_rate=6e-4, 
    device_type=device_type,
)
if master_process:
    raw_model.log_optimizer_info(optimizer)

# Train config
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    device_type=device_type,
    ddp=ddp,
    ddp_rank=ddp_rank,
    ddp_world_size=ddp_world_size,
    master_process=master_process,
    log_dir=log_dir,
    max_steps=max_steps,
    max_lr=max_lr,
    min_lr=min_lr,
    warmup_steps=warmup_steps,
    eval_interval=250,
    checkpoint_interval=5000,
    val_loss_steps=20,
    grad_accum_steps=grad_accum_steps,
)

trainer.train()