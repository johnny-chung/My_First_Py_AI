
import torch
import torch.nn as nn

# Step 3: Define a neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 512)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()  # Use Tanh activation for the final layer
        self.fc2 = nn.Linear(512, 256)
        self.batch_norm = nn.BatchNorm1d(512)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.05)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(256, 64)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = self.tanh(x)  # Use Tanh activation for the final layer
        x = self.dropout(x)
        x = self.fc3(x)
        return x.view(-1, 8, 8)
