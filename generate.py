#!/usr/bin/env python3
"""
generate.py
-----------
Use a trained MusicTransformer to generate a new piano piece.

Pipeline:
- Load tokenized MAESTRO subset (e.g. Rachmaninoff only) from DATA_ROOT
- Load tokenizer.pkl (same tokenizer used in prep_maestro.py)
- Load trained model checkpoint (best_model.pt)
- Pick a seed piece from the training set
- Autoregressively sample new tokens
- Decode tokens back to MIDI with the tokenizer
- Save to OUTPUT_MIDI
"""

import json
import random
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ================================================================
#                            CONFIG
# ================================================================

DATA_ROOT = "./data/maestro_rachmaninoff"   # created by prep_maestro.py
CHECKPOINT_PATH = "./checkpoints/maestro_rachmaninoff_small/best_model.pt"

OUTPUT_DIR = "./samples"
OUTPUT_MIDI = "rachmaninoff_sample_02.mid"

# Generation hyperparameters
SEED = 1234           # RNG seed (does not affect training)
SEED_LEN = 1024        # number of tokens from the seed piece to keep
GEN_TOKENS = 256      # number of *new* tokens to generate

TEMPERATURE = 0.7
TOP_K = 10
TOP_P = 0.8

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
        # Upper triangular with -inf above diagonal -> causal
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] int64 token ids
        returns: [B, T, vocab_size] logits
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        h = self.token_emb(x) + self.pos_emb(pos)            # [B, T, d_model]

        mask = self._causal_mask(T, x.device)
        h = self.transformer(h, mask=mask)                   # [B, T, d_model]
        h = self.ln(h)
        logits = self.head(h)                                # [B, T, vocab_size]
        return logits

# ================================================================

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    logits: [vocab_size]
    """
    logits = logits.clone()

    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits[logits < min_values] = -float("inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("inf")

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> int:
    logits = logits / max(temperature, 1e-6)
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1)
    return int(token_id.item())

# ================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_root = Path(DATA_ROOT).resolve()
    ckpt_path = Path(CHECKPOINT_PATH).resolve()
    out_dir = Path(OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_midi = out_dir / OUTPUT_MIDI

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab_size = ckpt["vocab_size"]
    config = ckpt["config"]

    print(f"Checkpoint vocab_size = {vocab_size}")
    print(f"Model config from checkpoint: {config}")

    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        max_seq_len=config["seq_len"],
        dropout=0.1, # This MUST match training
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer_path = data_root / "tokenizer.pkl"
    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    index_path = data_root / "index.json"
    splits_train_path = data_root / "splits" / "train.txt"
    print(f"Loading index from {index_path}...")
    index = json.loads(index_path.read_text())
    train_ids = [x.strip() for x in splits_train_path.read_text().splitlines() if x.strip()]
    if not train_ids:
        raise RuntimeError("No train IDs found in splits/train.txt")

    seed_pid = random.choice(train_ids)
    seed_meta = index[seed_pid]
    token_file = data_root / seed_meta["path"]
    print(f"Using seed piece: {seed_pid}")
    print(f"  title   = {seed_meta.get('title', 'Unknown')}")
    print(f"  composer= {seed_meta.get('composer', 'Unknown')}")
    print(f"  tokens  = {token_file}")

    seed_tokens = np.load(token_file)
    print(f"Seed piece token length: {len(seed_tokens)}")

    seed = seed_tokens[:SEED_LEN]
    tokens = seed.tolist()
    print(f"Using first {len(tokens)} tokens of seed as context.")

    max_seq_len = config["seq_len"]
    print(f"Model max_seq_len = {max_seq_len}")
    print(f"Generating {GEN_TOKENS} new tokens...")

    with torch.no_grad():
        for i in range(GEN_TOKENS):
            ctx = tokens[-max_seq_len:]
            x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

            logits = model(x)               # [1, T, vocab]
            next_logits = logits[0, -1, :]  # [vocab]

            next_id = sample_next_token(
                next_logits,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
            )
            tokens.append(next_id)

            if (i + 1) % 50 == 0 or i == GEN_TOKENS - 1:
                print(f"  generated {i+1}/{GEN_TOKENS} tokens")

    full_tokens = np.array(tokens, dtype=np.int32)
    print(f"Total generated sequence length: {len(full_tokens)}")

    print("Decoding generated tokens to MIDI...")
    try:
        score = tokenizer.decode(full_tokens.tolist())
    except TypeError:
        score = tokenizer.decode([full_tokens.tolist()])

    print(f"Saving MIDI to {out_midi}...")
    try:
        score.dump_midi(str(out_midi))
    except AttributeError:
        try:
            midi_obj = score.to_midi()
            midi_obj.dump(str(out_midi))
        except Exception as e:
            print("[ERROR] Failed to write MIDI from decoded score:", repr(e))
            return

    print("Done.")


if __name__ == "__main__":
    main()