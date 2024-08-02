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
