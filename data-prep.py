# data preparation and sampling

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

with open('training_text/text.txt', 'r') as f :
    raw_text = f.read()
    print("Total length of words: ", len(raw_text.split(' ')))


encoded_text = tokenizer.encode(raw_text)

# creating input-target predictions
# to make sure the context size iterates over the entire dataset, 
# we will be making the datasets using torch's functions ayyyy

import torch
from torch.utils.data import Dataset, DataLoader


# inherited from torch's built in gptdatasetv1 class
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
