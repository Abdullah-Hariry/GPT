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

data = torch.tensor(encode(text), dtype=torch.long) # puts data in

# Validation
# Splits data into 9:1 ratio 9 being the data the transformer will train on and 1 being the data it will use to validate its results

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
# training set (x,y,z) when the input x is the context, y is the target, when the context is x,y the target is z
block_size = 8
print(train_data[:block_size+1])

x = train_data[:block_size] # [..., [8]]
y = train_data[1:block_size+1] # [[1] ... [9]]

# for t in range(block_size):
  #  context = x[:t+1] # t=1 context = [..., [2]] in the set of [..., [8]]
  #  target = y[t] # t=1, target = [[2]] in the set of [[1]... [9]]

batch_size = 4 # How many independent sequences will we process in parallel

torch.manual_seed(1337)
"""
get_batch() samples a batch by
1. Picking 4 random spots in the text
2. Cutting 8 characters at each spot we call this x (context)
3. Cutting 8 characters at the same spot but shifted by 1 this is y (target)
4. Stacking these 4 chunks into a single tensor
"""
def get_batch(split):
    data = train_data if split == 'train' else val_data # train_data = data[:n], hence data = data[:n] n being 90%
    ix = torch.randint(len(data) - block_size, (batch_size,)) # ix = (length(data[:n]) - 8
    x = torch.stack([data[i:i+block_size] for i in ix]) # x = []
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # y = []
    return x, y
xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('------')

for b in range(batch_size): # batch dimension
    for t in range(block_size): #time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")
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

m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print(out.shape)

