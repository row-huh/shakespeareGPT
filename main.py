import torch

# data preparation and sampling
# data preparation and sampling

import tiktoken
from data_prep import GPTDatasetV1, create_dataloader_v1

tokenizer = tiktoken.get_encoding("gpt2")

# fetch raw text from training_text/text.txt
with open('training_text/text.txt', 'r') as f :
    raw_text = f.read()
    print("Total length of words: ", len(raw_text.split(' ')))




# setup embedding layer (relative positional embeddings)
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

dataloader = create_dataloader_v1(
    raw_text, batch_size=4, max_length=6, stride=1, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs :\n", inputs)
print("\n Inputs SHape:\n", inputs)

# convert into relative positional embeddings
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


# embedding layers (absolute positional embeddings)
context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# attention mechanisms

# llm architecture

# training loop

# model evaluation

# saving weights