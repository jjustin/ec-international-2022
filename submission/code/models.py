import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class PenneModel(nn.Module):

    def __init__(self):
        super(PenneModel, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(3, 10, 4, stride=2, padding=1, bias=False),
            nn.MaxPool2d(10, 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(10, 10 * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(10 * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(10 * 2, 10 * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(10 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(360, 180, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(180, 90, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(90, 45, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(45, 4, bias=False),

            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class AnelliModel(nn.Module):

    def __init__(self):
        super(AnelliModel, self).__init__()
        self.convolution = nn.Sequential(

            nn.Conv2d(3, 10, 4, stride=2, padding=1, bias=False),
            nn.MaxPool2d(10, 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(10, 10 * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(10 * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(10 * 2, 10 * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(10 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(

            nn.Linear(360, 180, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(180, 90, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(90, 45, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(45, 4, bias=False),

            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        conv = self.convolution(input)
        combined = torch.cat([conv, input1])
        return self.linear(combined)


class FettuccineModel(nn.Module):

    def __init__(self):
        super(FettuccineModel, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RotiniModel(nn.Module):

    def __init__(self):
        super(RotiniModel, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(3, 8, 2, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.RNN(8, 20, 2),

            nn.Flatten(),

            nn.Linear((32 * 32 * 8), 4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
