import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.1
BATCH_SIZE = 32
EPOCH = 12

# make some data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(x.size()))

# make tensor dataset
dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# network design
class FitNet(torch.nn.Module):
    def __init__(self):
        super(FitNet, self).__init__()
        self.hiddenLayer = torch.nn.Linear(1, 20)
        self.outputLayer = torch.nn.Linear(20, 1)

    def forward(self, inputFeature):
        featureAfterHidden = F.relu(self.hiddenLayer(inputFeature))
        output = self.outputLayer(featureAfterHidden)
        return output


if __name__ == '__main__':
    fitterSGD = FitNet()
    fitterMomentum = FitNet()
    fitterRMSprop = FitNet()
    fitterAdam = FitNet()
    fitterNets = [fitterSGD, fitterMomentum, fitterRMSprop, fitterAdam]

    optimizerSGD = torch.optim.SGD(fitterSGD.parameters(), lr=LR)
    optimizerMomentum = torch.optim.SGD(fitterMomentum.parameters(), lr=LR, momentum=0.8)
    optimizerRMSprop = torch.optim.RMSprop(fitterRMSprop.parameters(), lr=LR, alpha=0.9)
    optimizerAdam = torch.optim.Adam(fitterAdam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [optimizerSGD, optimizerMomentum, optimizerRMSprop, optimizerAdam]

    lossFunction = torch.nn.MSELoss()
    lossHistorys = [[], [], [], []]

    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (xBatch, yBatch) in enumerate(loader):
            for fitterNet, optimizer, lossHistory in zip(fitterNets, optimizers, lossHistorys):
                output = fitterNet(xBatch)
                loss = lossFunction(output, yBatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lossHistory.append(loss.data.numpy())

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for index, lossHistory in enumerate(lossHistorys):
        plt.plot(lossHistory, label = labels[index])

plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(0, 0.2)
plt.show()
