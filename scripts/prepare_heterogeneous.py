import os
import pickle
import numpy as np
import re 
SPECIAL_TOKENS = [
    '[TEXT_START]', '[TEXT_END]',
    '[TEMP_START]', '[TEMP_END]',
    '[PRES_START]', '[PRES_END]',
    '[HR_START]',   '[HR_END]',
    '[SPO2_START]', '[SPO2_END]',
    '[STATUS_START]','[STATUS_END]',
]
with open('data/input_heterogeneous.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"length of dataset in characters: {len(text):,}")
chars = sorted(list(set(text)))
vocab = chars + SPECIAL_TOKENS
vocab_size = len(vocab)
print("all chars:", ''.join(chars))
print("special tokens:", SPECIAL_TOKENS)
print(f"vocab size: {vocab_size}")
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
escaped_tokens = [re.escape(t) for t in SPECIAL_TOKENS]
pattern = "(" + "|".join(escaped_tokens) + "|.)"
def encode(s):
    tokens = re.findall(pattern, s, re.DOTALL)
    return [stoi[t] for t in tokens]
def decode(l):
    return ''.join([itos[i] for i in l])
n = len(text)
train_data = text[:int(n*0.9)]
val_data = text[int(n*0.9):]
print("\nEncoding training data... (This will be much faster now)")
train_ids = encode(train_data)
print("Encoding validation data...")
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train_heterogeneous.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val_heterogeneous.bin'))
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta_heterogeneous.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print("\nHeterogeneous data preparation complete.")