from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

def get_data():
    return load_dataset("HuggingFaceH4/ultrachat_200k")


def get_processed_data():
    ds = get_data()
    data = []
    for conversations in ds['train_sft']:
        dialogue = ""
        for conversation in conversations['messages']:
            speaker = "User" if conversation['role'] == 'user' else "Bot"
            dialogue += f"{speaker}: {conversation['content']}\n"
        data.append(dialogue)
    return data


class TextData(Dataset):
    def __init__(self, tokenizer, max_len:int=128):
        self.text = get_processed_data()
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


def get_huggingface_dataset_loader(tokenizer, max_len, batch_size, shuflle=True):
    train_dataset = TextData(tokenizer, max_len)
    val_dataset = TextData(tokenizer, max_len)

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
    train_path = 'train_dataset.csv'
    val_path = 'val_dataset.csv'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_huggingface_dataset_loader(tokenizer, 32, 2)

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
