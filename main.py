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


dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print("First Batch:\n", first_batch)
second_batch = next(data_iter)
print("Second Batch:\n", second_batch)


# attention mechanisms

# llm architecture

# training loop

# model evaluation

# saving weights