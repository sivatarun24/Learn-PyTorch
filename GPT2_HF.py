
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tiktoken
import torch
from torch import nn
from torch.nn import functional as F

model_hf = GPT2LMHeadModel.from_pretrained('gpt2')

sd_hf = model_hf.state_dict()
# for k, v in sd_hf.items():
#     print(k, v)


max_return_sequences = 5
max_length = 30
starting_text = "Hello, I'm a language model,"
device = "cpu"

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(starting_text)
tokens = torch.tensor(tokens)
tokens = tokens.unsqueeze(0).repeat(max_return_sequences, 1)
x = tokens.to(device)

model_hf.eval()
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model_hf(x)
        logits = logits.logits
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=50)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, 1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(max_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(decoded)
