import torch.nn as nn
import torch

# model = nn.Linear(in_features=1, out_features=1,bias=True)
model = nn.Linear(1, 1) #4번과 같은 라인

class SingleLayer(nn.Module):
    def __init__(self, inputs):
        super(SingleLayer,self).__init__
        self.layer == nn.Linear(inputs, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layers3 = nn.Sequential(
            nn.Linear(in_features=30*5*5, out_features=10, bias=True),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = x.view(x.shape[0], -1)
        x = self.layers3(x)
        return x

def main():
        
    model = nn.Linear(in_features=1, out_features=1, bias=True)
    model = nn.Linear(1,1)
    print(model.weight)
    print(model.bias)
    model = MLP()
    print(list(model.children()))

if __name__ == '__main__':
    main()