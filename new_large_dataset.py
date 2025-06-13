import torch
import ast
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from typing import Dict, List
from sklearn.model_selection import train_test_split


paths = ['qa_Appliances.json', 'qa_Arts_Crafts_and_Sewing.json', 'qa_Automotive.json', 'qa_Sports_and_Outdoors.json']


def get_conversations(paths) -> Dict:
    data = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = ast.literal_eval(line)
                data.append(obj)
    df = pd.DataFrame(data)

    conversations = {}
    idx = 0  # unique conversation id
    for _, row in df.iterrows():
        question = row.get('question', '').strip()
        answer = row.get('answer', '').strip()
        if not question or not answer:
            continue
        conversations[f'conv_{idx}'] = [question, answer]
        idx += 1

    return conversations


def get_processed_data(paths, split=True, test_size=0.1, random_state=42)->List:
    processed_data = []
    conversations = get_conversations(paths)
    for conv in conversations.values():
        dialogue = ""
        for i, msg in enumerate(conv):
            speaker = "User" if i % 2 == 0 else "Bot"
            dialogue += f"{speaker}: {msg.strip()}\n"
        processed_data.append(dialogue)
    if split:
        return train_test_split(processed_data, test_size=test_size, random_state=random_state)
    else:
        return processed_data


class TextData(Dataset):
    def __init__(self, text, tokenizer, max_len:int=128):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = self.text[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        targets = input_ids.clone()
        targets[:-1] = input_ids[1:]
        targets[-1] = self.tokenizer.pad_token_id

        return {
            'input_ids':input_ids,
            'targets': targets,
            'attention_mask': attention_mask, 
        }
    

def decode_token_ids(token_ids, tokenizer):
    if torch.is_tensor(token_ids):
        token_ids = token_ids.tolist()

    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return decoded_text


def get_loader(paths, tokenizer, max_len, batch_size, shuflle=True):
    train_text, val_text = get_processed_data(paths)
    train_dataset = TextData(train_text, tokenizer, max_len)
    val_dataset = TextData(val_text, tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=shuflle,
        num_workers=2, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=shuflle,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader



if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_loader(paths[:1], tokenizer, 128, 2)

    train_len = len(list(train_loader))
    val_len = len(list(val_loader))

    print("train_len: ", train_len)
    print("val_len: ", val_len)

    for batch in train_loader:
        input_ids = batch['input_ids']
        targets = batch['targets']
        attention_mask = batch['attention_mask']

        print("input_ids: ", input_ids)
        print("targets: ", targets)
        print("attention_mask: ", attention_mask)
        break

    decoded_text = decode_token_ids(input_ids[0], tokenizer)
    print(decoded_text)





