import torch.nn as nn
import torch


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
