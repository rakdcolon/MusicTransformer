# Piano Transformer on MAESTRO

This project trains and runs a Transformer model for classical solo-piano music on the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset.  
It:

- Preprocesses MAESTRO MIDI files into token sequences (REMI via `miditok`)
- Creates train/val/test splits
- Loads a trained Transformer checkpoint
- Generates new piano MIDI from a seed piece

---

## 1. Setup

### 1.1. Clone & enter the project
```bash
git clone https://github.com/rakdcolon/MusicTransformer
cd MusicTransformer
```
### 1.2. Python & virtualenv

Requires **Python 3.12**.
```bash
python3 -m venv .venv
source .venv/bin/activate # on Windows: .venv\Scripts\activate
```
### 1.3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
---

## 2. Dataset: MAESTRO

1. Download the MAESTRO dataset (v3) from the official source linked above.
2. Extract it so that you have a structure like:

   ```txt
   data/
     maestro/
       metadata.csv
       2018/
       2019/
       ...
   ```

   Adjust the folder name as needed. The important part is that `metadata.csv` lives under `MAESTRO_ROOT` used by the prep script. Note that the dataset might download the CSV file under a different name, please rename the file to `metadata.csv`.

---

## 3. Data preparation

Main script: `prep_maestro.py`
- Reads `metadata.csv` under `MAESTRO_ROOT`
- Optionally filters by composer (e.g., Chopin-only subset)
- Tokenizes each MIDI with REMI (`miditok`)
- Saves tokens as `.npy` files under `OUT_ROOT/tokens/`
- Saves:
  - `OUT_ROOT/tokenizer.pkl` (tokenizer + vocab)
  - `OUT_ROOT/index.json` (piece metadata and token paths)
  - `OUT_ROOT/splits/{train,val,test}.txt` (IDs)

### 3.1. Configure

Open `prep_maestro.py` and adjust the config block at the top:

- `MAESTRO_ROOT` — path to the MAESTRO folder containing `metadata.csv`
- `OUT_ROOT` — where tokenized data and splits will be written
- `COMPOSERS` — comma-separated string of composers to keep (e.g. `"Chopin"`), or empty for all
- `MIN_TOKENS`, `TRAIN_FRAC`, `VAL_FRAC`, `SEED` as desired

### 3.2. Run
```bash
python prep_maestro.py
```
After it finishes, you should have something like:
```txt
data/
  maestro_chopin/
    tokens/
    splits/
      train.txt
      val.txt
      test.txt
    index.json
    tokenizer.pkl
```
---

## 4. Training

Use `OUT_ROOT` (e.g. `./data/maestro_chopin`) as your dataset root for training your Music Transformer:

- Reads token sequences from `tokens/`
- Uses `splits/train.txt`, `splits/val.txt`, `splits/test.txt` for splits
- Saves a checkpoint that includes:
  - Model state dict
  - Vocabulary size
  - Model config (d_model, n_layers, n_heads, d_ff, seq_len, etc.)

---

## 5. Generation

Main script: `generate.py`
- Loads tokenized dataset from `DATA_ROOT`
- Loads `tokenizer.pkl` from that directory
- Loads a trained Transformer checkpoint from `CHECKPOINT_PATH`
- Picks a random seed piece from the training split
- Generates additional tokens autoregressively
- Decodes tokens back to MIDI and saves it

### 5.1. Configure

At the top of `generate.py`, set:

- `DATA_ROOT` — same as `OUT_ROOT` from the prep script (e.g. `./data/maestro_chopin`)
- `CHECKPOINT_PATH` — path to your trained model (e.g. `./checkpoints/.../best_model.pt`)
- `OUTPUT_DIR` and `OUTPUT_MIDI` — where the generated MIDI will be written
- Generation hyperparameters: `SEED`, `SEED_LEN`, `GEN_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`

### 5.2. Run
```bash
python generate.py
```
The generated MIDI file will appear under `OUTPUT_DIR/OUTPUT_MIDI`.

---

## 6. Script order summary

1. Create and activate virtualenv  
2. `pip install -r requirements.txt`  
3. Download and extract MAESTRO under `MAESTRO_ROOT`
4. Configure and run `prep_maestro.py`
5. Train the Transformer model on the prepared data
6. Configure and run `generate.py` to produce MIDI samples