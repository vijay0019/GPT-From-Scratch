import os
import time
import logging
import math
import socket
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
from hellaswag import get_most_likely_row
from logger import CSVLogger

logger = logging.getLogger(__name__)

def setup_ddp():
    """Set up distributed data parallel training."""
    if "SLURM_PROCID" in os.environ:
        ddp = True

        ddp_rank = int(os.environ["SLURM_PROCID"])
        ddp_local_rank = int(os.environ["SLURM_LOCALID"])
        ddp_world_size = int(os.environ["SLURM_NTASKS"])

        # Get master address from Slurm
        if "SLURM_NODELIST" in os.environ:
            hostnames = os.popen(
                "scontrol show hostname " + os.environ["SLURM_NODELIST"]
            ).read().split()
            master_addr = hostnames[0]
        else:
            master_addr = socket.gethostname()

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

        torch.cuda.set_device(ddp_local_rank)
        device = f"cuda:{ddp_local_rank}"

        dist.init_process_group(
            backend="nccl",
            rank=ddp_rank,
            world_size=ddp_world_size,
            device_id=torch.device(f"cuda:{ddp_local_rank}"),
        )
        master_process = ddp_rank == 0
    else:
        ddp = False
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process

def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    """Learning rate schedule with linear warmup and cosine decay."""
    
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

class Trainer:
    """
    Trainer class for GPT model training.
    """
    
    def __init__(self, model, optimizer,
        train_loader, val_loader,
        device, device_type,
        ddp=False, ddp_rank=0, ddp_world_size=1, master_process=True,
        log_dir="log",
        max_steps=19073, max_lr=6e-4, min_lr=None, warmup_steps=715,
        eval_interval=250, checkpoint_interval=5000, 
        val_loss_steps=20, grad_accum_steps=1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.device_type = device_type
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.master_process = master_process
        self.log_dir = log_dir
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else max_lr * 0.1
        self.warmup_steps = warmup_steps
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.val_loss_steps = val_loss_steps
        self.grad_accum_steps = grad_accum_steps
        
        # CSV logger
        self.csv_logger = CSVLogger(log_dir=log_dir)
        os.makedirs(log_dir, exist_ok=True)
    
    def evaluate_validation_loss(self, step):
        """Evaluate validation loss."""
        self.model.eval()
        self.val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(self.val_loss_steps):
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                loss = loss / self.val_loss_steps
                val_loss_accum += loss.detach()
        
        val_loss = val_loss_accum.item()
        
        if self.master_process:
            logger.info(f"validation loss: {val_loss:.4f}")
            self.csv_logger.log(step, 'val', val_loss)
        
        return val_loss
    
    def evaluate_hellaswag(self, step):
        """Evaluate on HellaSwag dataset."""
        try:
            from hellaswag import iterate_examples, render_example
        except Exception as e:
            logger.warning(f"HellaSwag evaluation not available: {e}. Skipping.")
            return
        
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(self.device)
            mask = mask.to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        
        acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0
        
        # if self.master_process:
        logger.info(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        self.csv_logger.log(step, 'hella', acc_norm)
        
        return acc_norm
    
    def save_checkpoint(self, step, val_loss, raw_model):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step,
            'val_loss': val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train_step(self, step):
        """Perform one training step."""
        t0 = time.time()
        last_step = (step == self.max_steps - 1)
        
        # Evaluate validation loss and save checkpoint
        if (step % self.eval_interval == 0) or last_step:
            if self.master_process:
                val_loss = self.evaluate_validation_loss(step)
                if step>0:
                    raw_model = self.model.module if self.ddp else self.model
                    self.save_checkpoint(step, val_loss, raw_model)

        # Evaluate HellaSwag
        if (step % self.eval_interval == 0) or last_step:
            if self.master_process:
                try:
                    self.evaluate_hellaswag(step)
                except Exception as e:
                    if self.master_process:
                        logger.warning(f"HellaSwag evaluation failed: {e}")
            if self.ddp:
                dist.barrier()
        
        # Training step
        self.model.train()
        self.optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(self.grad_accum_steps):
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            
            if self.ddp:
                self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)
            
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)
            
            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
        if self.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update LR
        lr = get_lr(step, self.max_lr, self.min_lr, self.warmup_steps, self.max_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.optimizer.step()
        
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps * self.ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if self.master_process:
            logger.info(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | "
                f"lr {lr:.4e} | norm: {norm:.4f} | "
                f"dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            self.csv_logger.log(
                step, 'train', loss_accum.item(),
                lr=lr, norm=norm.item(), dt_ms=dt*1000, tokens_per_sec=tokens_per_sec
            )
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.max_steps} steps")
        
        for step in range(self.max_steps):
            self.train_step(step)
        
        logger.info("Training completed")
        
        if self.ddp:
            destroy_process_group()