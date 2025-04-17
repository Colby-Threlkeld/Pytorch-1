from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root='data', download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
#1,28,28 - classes 0-9

#Image Classifier Neural Network Setup
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequenctial(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReuLU(),
            nn.Conv2d(32,64, (3,3)),
            nn.ReLU(),
            nn.Cinv2s(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.linear(64*(28-6)*(28-6), 10)
        )
    def forward(self, x):
        return self.model(x)

