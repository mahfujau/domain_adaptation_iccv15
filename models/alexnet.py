import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    """AlexNet encoder model for our Model."""

    def __init__(self):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # fc6
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # fc7
            nn.Linear(4096, 4096),
        )

    def forward(self, x):
        # feature
        x = self.features(x)        
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x 


class ClassClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ClassClassifier, self).__init__()
        # fc8
        self.fc8 = nn.Sequential(
            # fc8
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.fc8(F.relu(x))
        return x

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.fcD = nn.Sequential(
            # fcD
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.fcD(F.relu(x))
        return x