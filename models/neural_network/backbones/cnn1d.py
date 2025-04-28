import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, stride=2, padding=1)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=7, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.2))
        self.layer4 = nn.Flatten()
        self.layer5 = nn.Sequential(
            nn.Linear(25600, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # x = self.relu(x)
        # x = torch.mean(x, dim=2)
        # x = self.fc(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc(out)
        return out


