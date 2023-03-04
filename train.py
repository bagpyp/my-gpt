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
