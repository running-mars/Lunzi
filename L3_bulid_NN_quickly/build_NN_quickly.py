"""
build a neural network quickly
"""
import torch
import torch.nn.functional as F


class SimpleNet(torch.nn.Module):
    def __init__(self, numFeatures, numHidden, numOutput):
        super(SimpleNet, self).__init__()
        self.hiddenLayer = torch.nn.Linear(numFeatures, numHidden)
        self.outputLayer = torch.nn.Linear(numHidden, numOutput)

    def forward(self, inputFeature):
        featureAfterHidden = F.elu(self.hiddenLayer(inputFeature))
        outputResult = self.outputLayer(featureAfterHidden)
        return outputResult


simpler = SimpleNet(numFeatures=1, numHidden=10, numOutput=1)
print(simpler)

quicker = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)
print(quicker)
