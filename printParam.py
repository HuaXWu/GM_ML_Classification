from torch import nn
from torchsummary import summary

"""
    print net params
    author: wuhx
    data: 20221206
"""
model = nn.Sequential(nn.Flatten(),
                    nn.Linear(7500, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 2))

summary(model, input_size=(3, 50, 50), batch_size=16)