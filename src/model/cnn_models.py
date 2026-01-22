import torch.nn as nn
import torch

class CNN_Medium(nn.Module):
    def __init__(self, history_length=200, vocab_size=512, embedding_dim=32):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 8 bits + direction = 2^9 = 512
            embedding_dim=embedding_dim  # 32
        )

        # CNN layers
        self.conv1 = nn.Conv1d(embedding_dim, 32, 1)  # 32 input channels now!
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.AvgPool1d(4)

        self.fc1 = nn.Linear(64 * (history_length // 4), 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, 200] - indices (0-511)

        # Embedding lookup
        x = self.embedding(x)  # [batch, 200, 32]

        # Transpose for Conv1d
        x = x.transpose(1, 2)  # [batch, 32, 200]

        # CNN
        x = self.relu(self.conv1(x))  # [batch, 32, 200]
        x = self.relu(self.conv2(x))  # [batch, 64, 200]
        x = self.pool(x)              # [batch, 64, 50]

        # Flatten
        x = x.view(x.size(0), -1)     # [batch, 3200]

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x

#CNN_Ultralight
class CNN_Ultralight(nn.Module):
    def __init__(self, history_length=200, vocab_size=512, embedding_dim=32):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.conv1 = nn.Conv1d(embedding_dim, 64, 1)
        self.pool = nn.AvgPool1d(8)

        self.fc1 = nn.Linear(64 * (history_length // 8), 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, 500]
        x = self.embedding(x)        # [batch, 500, 32]
        x = x.transpose(1, 2)        # [batch, 32, 500]

        x = self.relu(self.conv1(x)) # [batch, 64, 500]
        x = self.pool(x)             # [batch, 64, 125]

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

#CNN_Light
class CNN_Light(nn.Module):
    def __init__(self, history_length=200, vocab_size=512, embedding_dim=16):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.conv1 = nn.Conv1d(embedding_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.AvgPool1d(4)

        self.fc1 = nn.Linear(64 * (history_length // 4), 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, 200]
        x = self.embedding(x)        # [batch, 200, 16]
        x = x.transpose(1, 2)        # [batch, 16, 200]

        x = self.relu(self.conv1(x)) # [batch, 32, 200]
        x = self.relu(self.conv2(x)) # [batch, 64, 200]
        x = self.pool(x)             # [batch, 64, 50]

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

#CNN_Heavy
class CNN_Heavy(nn.Module):
    def __init__(self, history_length=200, vocab_size=512, embedding_dim=32):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.conv1 = nn.Conv1d(embedding_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.AvgPool1d(4)

        self.fc1 = nn.Linear(128 * (history_length // 4), 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, 200]
        x = self.embedding(x)        # [batch, 200, 32]
        x = x.transpose(1, 2)        # [batch, 32, 200]

        x = self.relu(self.conv1(x)) # [batch, 32, 200]
        x = self.relu(self.conv2(x)) # [batch, 64, 200]
        x = self.relu(self.conv3(x)) # [batch, 128, 200]
        x = self.pool(x)             # [batch, 128, 50]

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

#CNN_Humongous
class CNN_Humongous(nn.Module):
    def __init__(self, history_length=200, vocab_size=512, embedding_dim=32):
        super().__init__()

        self.cnn_light = CNN_Light(history_length, vocab_size, embedding_dim=16)
        self.cnn_medium = CNN_Medium(history_length, vocab_size, embedding_dim)
        self.cnn_heavy = CNN_Heavy(history_length, vocab_size, embedding_dim)

        self.fusion = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, 200]

        light_pred = self.cnn_light(x)
        medium_pred = self.cnn_medium(x)
        heavy_pred = self.cnn_heavy(x)

        ensemble = torch.cat([light_pred, medium_pred, heavy_pred], dim=1)
        output = self.sigmoid(self.fusion(ensemble))

        return output