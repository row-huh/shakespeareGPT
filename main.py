import torch

# data preparation and sampling

import tiktoken
from data_prep import create_dataloader_v1
from architecture import GPTModel


GPT_CONFIG_124M = {
    "vocab_size":50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
# print(model.eval())



# getting text data

tokenizer = tiktoken.get_encoding('gpt2')

with open('training_text/text.txt', 'r') as f:
    raw_text = f.read()

total_characters = len(raw_text)
total_tokens = len(tokenizer.encode(raw_text))

print("Total characters: ", total_characters)
print("Total tokens: ", total_tokens) 


# training loop

# model evaluation

# saving weights