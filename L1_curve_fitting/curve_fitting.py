"""
fit a curve
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# prepare the data, x as input features and y as labels
x = np.arange(-2, 2, 0.1)
y = pow(x, 3) + pow(x, 2)
x = torch.unsqueeze(torch.FloatTensor(x), dim=1)    # tensor in pytorch need have 2 dimensions
y = torch.unsqueeze(torch.FloatTensor(y), dim=1)
x = Variable(x)
y = Variable(y)


class FitNet(torch.nn.Module):
    def __init__(self, featureWidth, hiddenWidth, outputWidth):
        super(FitNet, self).__init__()
        self.hiddenLayer = torch.nn.Linear(featureWidth, hiddenWidth)
        self.outputLayer = torch.nn.Linear(hiddenWidth, outputWidth)

    def forward(self, x):
        featureAfterHidden = F.relu(self.hiddenLayer(x))
        outputResult = self.outputLayer(featureAfterHidden)
        return outputResult


fitter = FitNet(featureWidth=1, hiddenWidth=50, outputWidth=1)
optimizer = torch.optim.SGD(fitter.parameters(), lr=0.1)
lossFunction = torch.nn.MSELoss()
print(fitter)
fittingFigure = plt.figure("Figure of fitting")
plt.ion()
plt.show()

for t in range(500):
    prediction = fitter(x)
    loss = lossFunction(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.01)

plt.ioff()
plt.show()
