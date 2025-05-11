import os
import torch
from model import GPT
from data_processing import decode_token_ids
from transformers import GPT2Tokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 64
num_heads = 1
num_layers = 1
batch_size = 4
block_size = 128
max_len = 128
dropout = 0.3

model_path = 'best_val_model.pt'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

m = GPT(embed_size, vocab_size, block_size, num_heads, num_layers, dropout=dropout)
if os.path.exists(model_path):
    m.load_state_dict(torch.load(model_path, map_location=device))

model = m.to(device)


user_tokens = tokenizer.encode("\nUser:", add_special_tokens=False)
user_tokens = torch.tensor(user_tokens, dtype=torch.long, device=device)


while True:
    input_text = input("User: ")
    if input_text.lower() == "exit":
        break

    PROMPT = f"User: {input_text}\nBot: "

    encoded_text = tokenizer(
        PROMPT, 
        padding=False,
        return_tensors='pt'
    )

    last_tokens = user_tokens.unsqueeze(0)

    context = encoded_text['input_ids'].to(device)

    answer = model.generate(context, 100, last_tokens)
    print("Bot:", decode_token_ids(answer[0], tokenizer).strip())
