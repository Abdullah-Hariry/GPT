import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(''.join(chars))
print(vocab_size)

# Tokenisation

# Create mapping from chars to integers
# Hello -> [52, 32, 11, 11, 23]

s_to_i = {ch:i for i, ch in enumerate(chars)}
i_to_s = {i:ch for i, ch in enumerate(chars)}

def encode(s):
    set = []
    for ch in s:
        set.append(s_to_i[ch])
    return set

def decode(s):
    return ''.join([i_to_s[i] for i in s])

data = torch.tensor(encode(text), dtype=torch.long)

# Validation
# Splits data into 9:1 ratio 9 being the data the transformer will train on and 1 being the data it will use to validate its results

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
print(train_data[:block_size+1])

x = train_data[:block_size]




