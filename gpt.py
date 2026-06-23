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




