import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        dropout=0.5
    ):
        super().__init__()

        # 1️ Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # 2️ Convolution layers (one per kernel size)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        # 3️ Dropout
        self.dropout = nn.Dropout(dropout)

        # 4️ Final classifier
        self.fc = nn.Linear(
            in_features=num_filters * len(kernel_sizes),
            out_features=1
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """

        # Embedding
        x = self.embedding(x)
        # x: [batch_size, seq_len, embedding_dim]

        # Conv1D expects channels first
        x = x.permute(0, 2, 1)
        # x: [batch_size, embedding_dim, seq_len]

        # Apply convolutions + ReLU + max pooling
        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            # c: [batch_size, num_filters, seq_len - k + 1]

            c = F.max_pool1d(c, kernel_size=c.shape[2])
            # c: [batch_size, num_filters, 1]

            conv_outputs.append(c.squeeze(2))
            # [batch_size, num_filters]

        # Concatenate all filter outputs
        x = torch.cat(conv_outputs, dim=1)
        # x: [batch_size, num_filters * len(kernel_sizes)]

        # Dropout
        x = self.dropout(x)

        # Final linear layer
        logits = self.fc(x).squeeze(1)
        # logits: [batch_size]

        # Sigmoid for probability
        probs = torch.sigmoid(logits)
        return probs
