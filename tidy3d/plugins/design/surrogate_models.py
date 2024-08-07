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

        self.fc1 = nn.Linear(
            int(pool2Out * cl2), l1
        )  # Adjust the size according to your input length
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
