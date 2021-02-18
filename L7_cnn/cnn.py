"""
a cnn lunzi, which is time consuming
in the future, transfer this to cuda
"""
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

trainData = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
print(trainData.train_data.size())
print(trainData.train_labels.size())
plt.imshow(trainData.train_data[0].numpy(), cmap='gray')
plt.title('%i' % trainData.train_labels[0])
plt.show()

loader = Data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)

testData = torchvision.datasets.MNIST(root='./mnist', train=False)
imageTest = torch.unsqueeze(testData.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
labelTest = testData.test_labels[:2000]


class NumRecognition(nn.Module):
    def __init__(self):
        super(NumRecognition, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.outputLayer = nn.Linear(32*7*7, 10)

    def forward(self, inputFeatures):
        featureAfterConv1 = self.conv1(inputFeatures)
        featureAfterConv2 = self.conv2(featureAfterConv1)
        featureFlatten = featureAfterConv2.view(featureAfterConv2.size(0), -1)
        output = self.outputLayer(featureFlatten)
        return output, featureFlatten


recognizer = NumRecognition()
print(recognizer)

optimizer = torch.optim.Adam(recognizer.parameters(), lr=LR)
lossFunction = nn.CrossEntropyLoss()

try:
    from sklearn.manifold import TSNE
    HAS_SK = True
except:
    HAS_SK = False
    print('Please install sklearn for layer visualization, via: pip install sklearn')
def plotLabels(lowDWeights, labels):
    plt.cla()
    xSet, ySet = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, label in zip(xSet, ySet, labels):
        colorBackground = cm.rainbow(int(255*label/9));
        plt.text(x, y, label, backgroundcolor=colorBackground, fontsize=9)
    plt.xlim(xSet.min(), xSet.max())
    plt.ylim(ySet.min(), ySet.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


plt.ion()

for epoch in range(EPOCH):
    for step, (imageBatch, labelBatch) in enumerate(loader):
        output = recognizer(imageBatch)[0]
        loss = lossFunction(output, labelBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            outputTest, lastLayer = recognizer(imageTest)
            labelPredict = torch.max(outputTest, 1)[1].data.numpy()
            accuracy = float((labelPredict == labelTest.data.numpy()).astype(int).sum())/float(labelTest.size(0))
            print 'Epoch: ', epoch, '\t|train loss: %.4f' % loss.data.numpy(), '\t|test accuracy: %.2f' % accuracy
            if HAS_SK:
                # visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plotOnly = 5000
                lowDWeights = tsne.fit_transform(lastLayer.data.numpy()[:plotOnly, :])
                labels = labelTest.numpy()[:plotOnly]
                plotLabels(lowDWeights, labels)

plt.ioff()

testOutput, _ = recognizer(imageTest[:10])
labelPredict = torch.max(testOutput, 1)[1].data.numpy()
print(labelPredict, 'Prediction Numbers')
print(labelTest[:10].numpy(), 'Real Numbers')