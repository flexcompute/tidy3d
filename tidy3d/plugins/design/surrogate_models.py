import torch.nn as nn


class BasicReducingANN(nn.Module):
    def __init__(self, input_size, init_neurons, dropout) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, init_neurons)
        self.fc2 = nn.Linear(init_neurons, int(init_neurons / 2))
        self.fcOut = nn.Linear(int(init_neurons / 2), 1)

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop(self.relu(self.fc1(x)))
        out = self.drop(self.relu(self.fc2(out)))
        out = self.fcOut(out)

        return out


class Basic1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=10, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(608, 128)  # Adjust the size according to your input length
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
