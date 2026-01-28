import torch


# data preparation and sampling
import tiktoken
from data_prep import create_dataloader_v1
from architecture import GPTModel
from util import *

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


# splitting into training and validation batches

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


# calculating loss before training
device = torch.device('cuda')
model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
    
print("Training loss: ", train_loss)
print("Validation loss: ", val_loss)



torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,  weight_decay=0.1
)
num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5,  eval_iter=5,
    start_context="Thou art the", tokenizer=tokenizer
)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_tokens("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text: \n", tokens_to_text(token_ids, tokenizer))

# saving weights
torch.save(model.state_dict(), 'model.pth')
