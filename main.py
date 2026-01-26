import torch


# data preparation and sampling
import tiktoken
from data_prep import create_dataloader_v1
from architecture import GPTModel
from util import calc_loss_loader, calc_loss_batch


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
    text = f.read()

total_characters = len(text)
total_tokens = len(tokenizer.encode(text))

print("Total characters: ", total_characters)
print("Total tokens: ", total_tokens) 


# training loop

train_ratio = 0.90
split_index = int(train_ratio * len(text))
train_data = text[:split_index]
val_data = text[split_index:]


train_loader  = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


print("Train Loader: ")
for x,y in train_loader:
    print(x.shape, y.shape)
    
print("\nValidation loader: ")
for x,y in val_loader:
    print(x.shape, y.shape)


device = torch.device('gpu')
model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
    
print("Training loss: ", train_loss)
print("Validation loss: ", val_loss)

# model evaluation

# saving weights