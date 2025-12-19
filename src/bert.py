import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification


class BertTeacher:
    def __init__(self, device):
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(
            "textattack/bert-base-uncased-imdb"
        )

        self.model = BertForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-imdb"
        ).to(device)

        self.model.eval() 

    def predict_proba(self, texts):
        """
        texts: list[str]
        returns: tensor of shape [batch_size]
                 positive sentiment probabilities
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=1)
        return probs[:, 1]  # positive class
