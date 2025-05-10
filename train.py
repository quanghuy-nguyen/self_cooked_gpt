import os
import torch
import torch.optim as optim
from transformers import GPT2Tokenizer
from model import GPT
from data_processing import get_loader, decode_token_ids
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 64
num_heads = 1
num_layers = 1
batch_size = 4
block_size = 128
max_len = 128
dropout = 0.3

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
    out = out.mean()

    return out


def train_model(train_loader, model, optimizer, scheduler):
    for i, batch in tqdm(enumerate(train_loader)):
        x = batch['input_ids']
        y = batch['targets']     
        attention_mask = batch['attention_mask']

        x = x.to(device)
        y = y.to(device)
        attention_mask = attention_mask.to(device)

        _, loss = model(x, y, attention_mask)

        if i % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step: {i}, train_loss = {loss}, lr = {lr}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    train_loader, val_loader = get_loader(train_path, 
                                          val_path, 
                                          tokenizer, 
                                          max_len=max_len, 
                                          batch_size=batch_size)
    
    m = GPT(embed_size, vocab_size, block_size, num_heads, num_layers, dropout=dropout)
    if os.path.exists(model_path):
        m.load_state_dict(torch.load(model_path, map_location=device))
    model = m.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10000, gamma=0.9)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_model(train_loader, model, optimizer, scheduler)
        val_loss = evaluate(val_loader, model)
        print(f"Epoch: {epoch}, val_loss = {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()


        



        

