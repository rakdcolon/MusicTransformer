#!/usr/bin/env python3
"""
train.py
--------
Minimal Trainer for the Music Transformer
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ================================================================
#                            CONFIG
# ================================================================

DATA_ROOT = "./data/maestro_chopin"
OUT_DIR = "./checkpoints/maestro_chopin_small"

SEQ_LEN = 1024
BATCH_SIZE = 8
EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

D_MODEL = 512
N_LAYERS = 8
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1

MULTIPLIER = 100

SEED = 42
LOG_EVERY = 50
EVAL_EVERY = 1
USE_AMP = True

# ================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MusicTokenDataset(Dataset):
    """
    Random-window dataset over token sequences.
    """

    def __init__(self, data_root, index, split_ids, seq_len):
        self.data_root = Path(data_root).resolve()
        self.index = index
        self.entries = [(pid, self.index[pid]["path"], self.index[pid]["length"])
                        for pid in split_ids]
        self.seq_len = seq_len
        self.multiplier = MULTIPLIER

    def __len__(self):
        return len(self.entries) * self.multiplier

    def __getitem__(self, idx):
        pid, token_rel, length = self.entries[idx % len(self.entries)]
        token_path = (self.data_root / token_rel).resolve()

        tokens = np.load(token_path)
        n = tokens.shape[0]

        if n <= self.seq_len:
            pad = self.seq_len - n
            tokens = np.pad(tokens, (0, pad), constant_values=0)
        else:
            start = np.random.randint(0, n - self.seq_len)
            tokens = tokens[start:start + self.seq_len]

        return torch.from_numpy(tokens).long()

# ================================================================

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _causal_mask(self, T, device):
        mask = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.token_emb(x) + self.pos_emb(pos)
        mask = self._causal_mask(T, x.device)

        h = self.transformer(h, mask=mask)
        h = self.ln(h)
        return self.head(h)

# ================================================================

def estimate_vocab_size(index, data_root):
    max_id = 0
    for meta in index.values():
        arr = np.load(Path(data_root) / meta["path"])
        max_id = max(max_id, int(arr.max()))
    return max_id + 1


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        logits = model(inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            label_smoothing=0.1,
            reduction="sum",
        )

        num_tokens = tgt.numel()
        loss_norm = loss / num_tokens

        loss_norm.backward()
        optimizer.step()

        total_loss += loss.item()
        total_tokens += num_tokens

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        batch = batch.to(device)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]

        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(inp)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    reduction="sum",
                )
        else:
            logits = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="sum",
            )

        total_loss += loss.item()
        total_tokens += tgt.numel()

    avg = total_loss / total_tokens
    ppl = math.exp(avg) if avg < 20 else float("inf")
    return avg, ppl

# ================================================================

def main():
    import pickle

    set_seed(SEED)

    data_root = Path(DATA_ROOT).resolve()
    index = json.load(open(data_root / "index.json", "r"))

    # Load tokenizer
    with open(data_root / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    try:
        vocab_size = tokenizer.vocab_size
    except AttributeError:
        all_ids = set()
        for voc in tokenizer.vocab:
            if isinstance(voc, dict):
                all_ids.update(voc.keys())
        vocab_size = max(all_ids) + 1

    print(f"Vocab size from tokenizer = {vocab_size}")

    out_dir = Path(OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading train/val splits…")
    train_ids = [x.strip() for x in open(data_root / "splits/train.txt")]
    val_ids = [x.strip() for x in open(data_root / "splits/val.txt")]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    torch.set_float32_matmul_precision("high")

    # Datasets
    train_ds = MusicTokenDataset(data_root, index, train_ids, SEQ_LEN)
    val_ds = MusicTokenDataset(data_root, index, val_ids, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=-1)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=-1)

    # Model
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

        train_loss = train_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
        print(f"Train loss: {train_loss:.4f}")

        if epoch % EVAL_EVERY == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device, USE_AMP)
            print(f"Val loss: {val_loss:.4f}  |  ppl: {val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": model.state_dict(),
                     "vocab_size": vocab_size,
                     "config": {
                         "d_model": D_MODEL,
                         "n_layers": N_LAYERS,
                         "n_heads": N_HEADS,
                         "d_ff": D_FF,
                         "seq_len": SEQ_LEN,
                     }},
                    out_dir / "best_model.pt",
                )
                print("Saved new best model.")

        torch.save(
            {"model": model.state_dict(),
             "vocab_size": vocab_size},
            out_dir / "last_model.pt"
        )

    print("Training complete!")


if __name__ == "__main__":
    main()
