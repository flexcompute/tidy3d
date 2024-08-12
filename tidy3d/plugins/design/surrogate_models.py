import torch
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
    def __init__(self, resolution, cl1, cl2, c_kernal, l1, dropout):
        super().__init__()
        stride = 1
        padding = int(c_kernal / 2)  # Exploiting round down from int()
        p_kernal = 2
        p_stride = 2
        p_padding = 0
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=cl1, kernel_size=c_kernal, stride=stride, padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=cl1, out_channels=cl2, kernel_size=c_kernal, stride=stride, padding=padding
        )
        self.pool = nn.MaxPool1d(kernel_size=p_kernal, stride=p_stride, padding=p_padding)

        # Calculate input size to fc1
        conv1Out = ((resolution - c_kernal + 2 * padding) / stride) + 1
        pool1Out = ((conv1Out - p_kernal) / p_stride) + 1
        conv2Out = ((pool1Out - c_kernal + 2 * padding) / stride) + 1
        pool2Out = ((conv2Out - p_kernal) / p_stride) + 1

        self.fc1 = nn.Linear(int(pool2Out * cl2), l1)
        self.fc2 = nn.Linear(l1, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ComplexCNN1DRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(
            256 * 6, 512
        )  # Adjust input dimension based on the input signal length
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class VariableANN(nn.Module):
    def __init__(self, input_size, neurons, dropout) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, neurons[0]))

        for i in range(1, len(neurons)):
            self.layers.append(nn.Linear(neurons[i - 1], neurons[i]))

        self.layers.append(nn.Linear(neurons[-1], 1))

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.drop(self.relu(layer(x)))

        x = self.layers[-1](x)

        return x


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state for the LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
