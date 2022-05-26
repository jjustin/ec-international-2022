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
    HISTORY_LEN = 10

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

            nn.Linear(360+AnelliModel.HISTORY_LEN, 200, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(200, 90, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(90, 45, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(45, 4, bias=False),

            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        conv = self.convolution(input1)
        combined = torch.cat([conv, input2], dim=1)
        return self.linear(combined)


class FettuccineModel(nn.Module):

    def __init__(self):
        super(FettuccineModel, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(3, 20, 5, stride=2, padding=1, bias=False),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(20, 64, 5, stride=3, dilation=1, padding=3, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 32, 3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Flatten(),

            nn.Linear(32, 16, bias=False),
            nn.ReLU(),

            nn.Linear(16, 4, bias=False),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)


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


class SpaghettiModel(nn.Module):

    def __init__(self):
        super(SpaghettiModel, self).__init__()
        self.main = nn.Sequential(

            nn.MaxPool2d(3, 2),
            nn.Conv2d(3, 8, 5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 64, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 32, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(32, 16, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(16, 8, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(8, 4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
