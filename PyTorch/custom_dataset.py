import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

from SimpleNN import SimpleNN
from CNN import MyCNN

import numpy as np
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, in_csv_file, out_csv_file, transform = None):
        self.input_moves = pd.read_csv(in_csv_file, header = None)
        self.output_moves = pd.read_csv(out_csv_file, header = None)
        self.transform = transform

    def __len__ (self):
        return len(self.input_moves)
    
    def __getitem__(self, idx):
        in_move = torch.tensor(self.input_moves.values[idx], dtype=torch.float32).view(1, 8, 8)
        out_move = torch.tensor(self.output_moves.values[idx], dtype=torch.float32)

        if self.transform:
            in_move = self.transform(in_move)
            out_move = self.transform(out_move)
        
        return in_move, out_move

# Step 2: Create data loaders
batch_size = 32
input_csv_path = 'input.csv'
output_csv_path = 'output.csv'

dataset = CustomDataset(input_csv_path, output_csv_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Step 4: Specify a loss function
# criterion = nn.MSELoss()
# Example with Smooth L1 Loss
criterion = nn.SmoothL1Loss()


model = MyCNN()

# Step 5: Choose an optimizer
learning_rate = 0.0001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Define a learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Step 6: Train the model
num_epochs = 20
epoch = 0
exit = False
while epoch < num_epochs and exit == False:
    for input_batch, output_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(input_batch)
        loss = criterion(predictions, output_batch)
        loss.backward()
        optimizer.step()

    #scheduler.step()
    if loss.item() < 0.3:
        exit = True
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    epoch += 1


torch.save(model.state_dict(), 'trained_reversi_model_2.pth')