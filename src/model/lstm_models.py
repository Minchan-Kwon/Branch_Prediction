import torch
import torch.nn as nn

#LSTM
class LSTM_Medium(nn.Module):
    def __init__(self, vocab_size=512, embedding_dim=32, hidden_size=200, num_layers=2):
        super().__init__()

        # Embedding layer 
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 512
            embedding_dim=embedding_dim  # 32
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,    # 32 (embedding output)
            hidden_size=hidden_size,     #
            num_layers=num_layers,       #
            batch_first=True
        )

        # FC layer
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, 200] - indices

        # Embedding
        x = self.embedding(x)  # [batch, 200, 32]

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, 200, 128]

        # Use last timestep
        last_out = lstm_out[:, -1, :]  # [batch, 128]

        # Prediction
        out = self.fc(last_out)
        out = self.sigmoid(out)

        return out