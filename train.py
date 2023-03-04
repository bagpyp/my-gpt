# %% load shakespeare
with open("tinyshakespeare.txt", encoding="utf-8") as file:
    text = file.read()
print("length of text: ", len(text))
print(text[:1000])

# %% generate vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size, " chars in vocab: ", "".join(chars))

# %% tokenizer (not "sub-word" like what gpt uses (tiktoken))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {v: k for k, v in stoi.items()}
encode = lambda s: [stoi[c] for c in s]  # encoder: str -> int[]
decode = lambda l: "".join([itos[i] for i in l])  # decoder: int[] -> str

print(encode("hi robbie!"))
print(decode(encode("hi robbie!")))

# %% encode entire text dataset and store as torch.Tensor
import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# %% split data into train/validation sets
n = int(0.9 * len(data))  # first 90% is training data
train_data = data[:n]
val_data = data[n:]

# %% block size and vlock size + 1
block_size = 8
print(train_data[:block_size + 1])
x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f'when context is {context}, the target is {target}')

# %% generating batches
batch_size = 4  # for parallel processing
block_size = 8  # maximum context length for predictions


def get_batch(split):
    # generates a small batch of data of inputs x and targets y
    batch_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(batch_data) - block_size, (batch_size,))
    x = torch.stack([batch_data[i:i + block_size] for i in ix])
    y = torch.stack([batch_data[i + 1:i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch('train')
print('inputs: \n', xb.shape, '\n', xb)  # our input to the transformer
print('targets: \n', yb.shape, '\n', yb)

# %% bigram
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# %% sample, only with bigram, not using history
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))


# %% create a PyTorch optimizer (3e-4 normally)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# %%
batch_size = 32
for steps in range(10000):

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())