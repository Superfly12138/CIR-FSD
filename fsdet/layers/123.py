import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(10, 16, 30, 32,device="cuda:0")
# batch, channel , height , width
print(x.shape)
class Net_1D(nn.Module):
    def __init__(self):
        super(Net_1D, self).__init__()
        self.layers = nn.Sequential(
            # nn.Conv1d(in_channels=16, out_channels=2, kernel_size=(3,3), stride=2, padding=1),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=(3,3), stride=2, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        output = self.layers(x)
        return  output

n = Net_1D()  # in_channel,out_channel,kennel,
print(x.shape)
y = n(x)
print(y.shape)
