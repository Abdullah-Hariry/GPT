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
# training set (x,y,z) when the input x is the context, y is the target, when the context is x,y the target is z
block_size = 8
print(train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]

batch_size = 4 # How many independent sequences will we process in parallel
#generate a small of batch of data of inputs x and targets y

torch.manual_seed(1337)
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randit(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
xb, yb = get_batch('train')
#bigram Language model

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table =  nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx) # (B,T,C)

        return logits

m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print(out.shape)

