import torch
import torch.nn as nn

class FCN_1D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCN_1D, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)  
        return x
