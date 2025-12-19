
import re
from collections import Counter


def tokenize(text):
    """
    Lowercase and split text into words.
    """
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

def build_vocab(texts, max_vocab_size=20000, min_freq=2):
    """
    Build word -> index vocabulary from training texts.
    """
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, freq in counter.most_common(max_vocab_size):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)

    return vocab

def encode(text, vocab, max_length=256):
    """
    Convert text to a fixed-length list of word indices.
    """
    tokens = tokenize(text)

    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    # truncate
    ids = ids[:max_length]

    # pad
    if len(ids) < max_length:
        ids += [vocab["<PAD>"]] * (max_length - len(ids))

    return ids
