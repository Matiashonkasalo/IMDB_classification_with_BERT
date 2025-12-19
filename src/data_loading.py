import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from text_processing import build_vocab, encode

def load_imdb():
    """
    Loads the IMDb dataset.
    Returns train and test splits.
    """
    dataset = load_dataset("imdb")
    return dataset["train"], dataset["test"]


def collate_fn(batch):
    texts = [item["text"] for item in batch]
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    return texts, input_ids, labels

def get_dataloader(dataset, batch_size=8, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


class IMDbDataset(Dataset):
    def __init__(self, data, vocab, max_length=256):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["text"]
        label = item["label"]

        input_ids = encode(text, self.vocab, self.max_length)

        return {
            "text": text,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float)
        }



if __name__ == "__main__":
    train_data, _ = load_imdb()

    # take a subset just for quick testing
    subset = train_data.select(range(1000))
    texts = [item["text"] for item in subset]

    # build vocab
    vocab = build_vocab(texts)

    # create dataset
    dataset = IMDbDataset(train_data, vocab)

    dataset = get_dataloader(dataset)

