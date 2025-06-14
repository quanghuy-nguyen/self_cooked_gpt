import os
import math
import torch
import torch.optim as optim
from transformers import GPT2Tokenizer
from model import GPT, GPTConfig
from data_processing import get_loader, decode_token_ids
from new_large_dataset import get_large_dataset_loader, paths
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 3e-4
num_epochs = 2
eval_iter = 2
train_path = 'train_dataset.csv'
val_path = 'val_dataset.csv'
model_path = 'best_val_model.pt'


@torch.no_grad()
def evaluate(eval_loader, model):
    model.eval()
    out = torch.zeros(eval_iter)
    for i in range(eval_iter):
        losses = torch.zeros(len(list(eval_loader)))
        for j, batch in enumerate(eval_loader):
            x = batch['input_ids']
            y = batch['targets']
            attention_mask = batch['attention_mask']

            x = x.to(device)
            y = y.to(device)
            attention_mask = attention_mask.to(device)

            _, loss = model(x, y, attention_mask)
            losses[j] = loss.item()
        out[i] = losses.mean()

    model.train()
    val_loss = out.mean().item()
    perplexity = math.exp(val_loss)

    return val_loss, perplexity


def train_model(train_loader, model, optimizer, scheduler, scaler):
    for i, batch in tqdm(enumerate(train_loader)):
        x = batch['input_ids']
        y = batch['targets']     
        attention_mask = batch['attention_mask']

        x = x.to(device)
        y = y.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(): # Make model run FP16 automaticaly
            _, loss = model(x, y, attention_mask)

        scaler.scale(loss).backward() # Using scale to scale the loss before backward() to avoid gradient underflow (too small loss)
        scaler.unscale_(optimizer) # Unscale the loss after scaling
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Using gradient cliping to avoid exploding gradients
        scaler.step(optimizer)
        scaler.update()

        if i % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step: {i}, train_loss = {loss}, lr = {lr}")

        scheduler.step()


def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    gpt_config = GPTConfig()

    # train_loader, val_loader = get_loader(train_path, 
    #                                       val_path, 
    #                                       tokenizer, 
    #                                       max_len=gpt_config.max_len, 
    #                                       batch_size=gpt_config.batch_size)

    train_loader, val_loader = get_large_dataset_loader(paths,
                                          tokenizer, 
                                          max_len=gpt_config.max_len, 
                                          batch_size=gpt_config.batch_size)
    print(f"Train len: {len(list(train_loader))}")
    print(f"Val len: {len(list(val_loader))}")
    
    m = GPT(gpt_config)
    if os.path.exists(model_path):
        m.load_state_dict(torch.load(model_path, map_location=device))
    model = m.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8000, gamma=0.9)
    scaler = GradScaler()

    best_val_loss = float('inf')
    wait = 0 # For early stopping
    patience = 3
    for epoch in range(num_epochs):
        train_model(train_loader, model, optimizer, scheduler, scaler)
        val_loss, perplexity = evaluate(val_loader, model)
        print(f"Epoch: {epoch} | Val loss = {val_loss} | Perplexity = {perplexity}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            wait = 0 # Reset wait to 0
        else:
            wait += 1
            if wait > patience:
                print("Overfitting!!!!!")
                break



if __name__ == "__main__":
    main()


        



        

