# create a lstm model to predict car movements
# input: current four wheels speed, xyz position, angles(sin, cos) *9
# output: next xyz position, angles(sin, cos) *5

import torch
import torch.nn as nn

class CarPredictor(nn.Module):
    def __init__(self, input_size=9, hidden_size=30, output_size=5, num_layers=2, dropout=0.2):
        super(CarPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out