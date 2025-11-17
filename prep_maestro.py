#!/usr/bin/env python3
"""
prep_maestro.py
---------------
Data preparation script for a classical piano Transformer on MAESTRO.

Pipeline:
- Reads MAESTRO metadata.csv
- Tokenizes each MIDI with miditok
- Saves tokens to .npy
- Writes index.json with metadata
- Creates train/val/test splits by piece
"""

# ============================== CONFIG ==============================

MAESTRO_ROOT = "./data/maestro"    # folder that contains metadata.csv or maestro-v*/ subfolder
OUT_ROOT = "./data/maestro_chopin" # where tokens/, index.json and splits/ will be written

COMPOSERS = "Chopin" # e.g. "Rachmaninoff", empty = all
MIN_TOKENS = 128     # drop pieces with fewer tokens

SEED = 24
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

# ===================================================================

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import pickle

from miditok import REMI, TokenizerConfig

def build_tokenizer() -> REMI:
    """
    Create a REMI tokenizer configuration meant for solo piano.
    """
    config = TokenizerConfig()

    # Solo piano range (A0–C8)
    config.pitch_range = (21, 109)
    config.num_velocity_bins = 32

    # Time resolution: 4 beats per bar, 8 ticks per beat = 32 positions/bar
    config.beat_res = {
        (0, 4): 8
    }

    config.use_programs = True
    config.one_token_stream_for_programs = True
    config.use_chords = False
    config.use_tempos = True
    config.use_time_signatures = True
    config.use_drums = False

    tokenizer = REMI(config)
    tokenizer.one_token_stream = True

    return tokenizer

def load_maestro_metadata(maestro_root: Path) -> tuple[DataFrame, Path]:
    """
    Load MAESTRO metadata.csv from the given root directory.
    Assumes metadata.csv is at <maestro_root>/metadata.csv or in a subdir.
    """
    metadata_path = maestro_root / "metadata.csv"

    if metadata_path is None:
        raise FileNotFoundError(f"Could not find metadata.csv in {maestro_root}.")

    df = pd.read_csv(metadata_path)
    return df, metadata_path.parent


def filter_composers(df: pd.DataFrame, composers: list[str] | None) -> pd.DataFrame:
    """
    Filter metadata rows by canonical_composer.
    """
    if not composers: # If 'composers' is None or empty, return df unchanged.
        return df

    comps_norm = [c.lower() for c in composers]

    def match(row): # Case-insensitive matching
        name = str(row.get("canonical_composer", "")).lower()
        return any(c in name for c in comps_norm)

    mask = df.apply(match, axis=1)
    return df[mask].reset_index(drop=True)


def tokenize_piece(midi_path: str, tokenizer: REMI) -> np.ndarray:
    """
    Tokenize a MIDI file with MidiTok REMI using the API.
    """
    try:
        tokens_obj = tokenizer.encode(midi_path)
    except Exception as e:
        print(f"[WARN] Tokenization failed for {midi_path}: {e}")
        raise Exception("Tokenization failed")

    # tokens_obj can be a TokSequence or a list[TokSequence] depending on settings
    if isinstance(tokens_obj, list):
        if len(tokens_obj) == 0:
            print(f"[WARN] No tokens in {midi_path}")
            raise Exception("No tokens in MIDI")
        seq = tokens_obj[0]   # for solo piano we expect one track
    else:
        seq = tokens_obj

    if not hasattr(seq, "ids"):
        print(f"[WARN] Unexpected token object format for {midi_path}: {type(seq)}")
        raise Exception("Unexpected token object format")

    ids = np.array(seq.ids, dtype=np.int32)
    return ids


def main():
    maestro_root = Path(MAESTRO_ROOT).expanduser().resolve()
    out_root = Path(OUT_ROOT).expanduser().resolve()
    out_tokens_dir = out_root / "tokens"
    out_splits_dir = out_root / "splits"
    out_root.mkdir(parents=True, exist_ok=True)
    out_tokens_dir.mkdir(parents=True, exist_ok=True)
    out_splits_dir.mkdir(parents=True, exist_ok=True)

    composers = [c.strip() for c in COMPOSERS.split(",") if c.strip()]

    print(f"Loading MAESTRO metadata from {maestro_root}...")
    df, midi_base = load_maestro_metadata(maestro_root)
    print(f"Loaded {len(df)} rows from metadata.")

    if composers:
        df = filter_composers(df, composers)
        print(f"Filtered to {len(df)} rows after composer filter: {composers}")

    tokenizer = build_tokenizer()
    print("Tokenizer ready (REMI configuration).")

    index = {}

    print("Tokenizing pieces...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        midi_rel_path = row["midi_filename"]
        composer = row.get("canonical_composer", row.get("composer", "Unknown"))
        piece_title = row.get("canonical_title", row.get("piece_title", "Unknown"))
        year = int(row.get("year", -1)) if not pd.isna(row.get("year", np.nan)) else -1

        midi_path = midi_base / midi_rel_path
        if not midi_path.exists():
            print(f"[WARN] MIDI file not found: {midi_path}")
            continue

        piece_id = f"maestro_{i:05d}"

        token_path = out_tokens_dir / f"{piece_id}.npy"
        if token_path.exists(): # already processed
            tokens = np.load(token_path)
            length = int(tokens.shape[0])
        else:
            tokens = tokenize_piece(str(midi_path), tokenizer)
            if tokens is None:
                continue

            if tokens.shape[0] < MIN_TOKENS:
                print(f"[INFO] Skipping {midi_path} (only {tokens.shape[0]} tokens)")
                continue

            np.save(token_path, tokens)
            length = int(tokens.shape[0])

        index[piece_id] = {
            "path": str(token_path.relative_to(out_root)),
            "composer": str(composer),
            "title": str(piece_title),
            "source": "maestro",
            "year": int(year),
            "length": length,
        }

    # Save tokenizer (including its vocabulary) for later decoding / generation
    tokenizer_path = out_root / "tokenizer.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Save index.json
    index_path = out_root / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Saved index.json with {len(index)} pieces at {index_path}")

    # Create train/val/test splits
    all_ids = list(index.keys())
    random.seed(SEED)
    random.shuffle(all_ids)

    n_total = len(all_ids)
    n_train = int(n_total * TRAIN_FRAC)
    n_val = int(n_total * VAL_FRAC)

    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train + n_val]
    test_ids = all_ids[n_train + n_val:]

    def write_split(name, ids):
        path = out_splits_dir / f"{name}.txt"
        with path.open("w", encoding="utf-8") as file:
            for pid in ids:
                file.write(pid + "\n")
        print(f"Saved {name} split with {len(ids)} ids to {path}")

    write_split("train", train_ids)
    write_split("val", val_ids)
    write_split("test", test_ids)

    print("Done.")


if __name__ == "__main__":
    main()