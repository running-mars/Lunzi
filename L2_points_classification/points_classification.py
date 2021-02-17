"""
classify two clusters of points
"""
import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 

# make some points for classification
numData = torch.ones(100, 2)    # 2 dimensional points
pointsGroup0 = torch.normal(4*numData, 2)
labelsGroup0 = torch.zeros(100)
pointsGroup1 = torch.normal(-4*numData, 2)
labelsGroup1 = torch.ones(100)
points = torch.cat((pointsGroup0, pointsGroup1), 0).type(torch.FloatTensor)
labels = torch.cat((labelsGroup0, labelsGroup1), 0).type(torch.LongTensor)


class ClassifyNet(torch.nn.Module):
    def __init__(self, numFeature, numHidden, numOutput):
        super(ClassifyNet, self).__init__()
        self.hiddenLayer = torch.nn.Linear(numFeature, numHidden)
        self.outputLayer = torch.nn.Linear(numHidden, numOutput)

    def forward(self, point):
        featureAfterHidden = F.relu(self.hiddenLayer(point))
        outputResult = self.outputLayer(featureAfterHidden)
        return outputResult


classifier = ClassifyNet(numFeature=2, numHidden=10, numOutput=2)
print(classifier)
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.002)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()   # turn on interactive mode

for t in range(150):
    predictLabels = classifier(points)
    loss = loss_func(predictLabels, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(predictLabels, 1)[1]
        predictedLabels = prediction.data.numpy()
        realLabels = labels.data.numpy()
        plt.scatter(points.data.numpy()[:, 0], points.data.numpy()[:, 1], c=predictedLabels, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((predictedLabels == realLabels).astype(int).sum()) / float(realLabels.size)
        plt.text(1.5, -4, 'Accuracy=%.5f' % accuracy, fontdict={'size': 12, 'color': 'red'})
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.3)

plt.ioff()  # turn off interactive mode
plt.show()
