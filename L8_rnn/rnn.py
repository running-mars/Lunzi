"""
rnn of number recognition
"""
import torch
import torch.utils.data as Data
from torch import nn
import torchvision.datasets as dataSets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

trainDataset = dataSets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

print(trainDataset.train_data.size())
print(trainDataset.train_labels.size())
plt.imshow(trainDataset.train_data[0].numpy(), cmap='gray')
plt.title("%i" % trainDataset.train_labels[0])
plt.show()

trainLoader = Data.DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True)

testDataset = dataSets.MNIST(
    root='./mnist/',
    train=False,
    transform=transforms.ToTensor()
)
testImage = testDataset.test_data.type(torch.FloatTensor)[:2000]/255.
testLabel = testDataset.test_labels.numpy()[:2000]


class NumRecognizer(nn.Module):
    def __init__(self):
        super(NumRecognizer, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, inputFeature):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(inputFeature, None)

        output = self.out(r_out[:, -1, :])
        return output


recognizer = NumRecognizer()
print(recognizer)

optimizer = torch.optim.Adam(recognizer.parameters(), lr=LR)
lossFunction = nn.CrossEntropyLoss()

# training
for epoch in range(EPOCH):
    for step, (imageBatch, labelBatch) in enumerate(trainLoader):
        imageBatch = imageBatch.view(-1, 28, 28)

        output = recognizer(imageBatch)
        loss = lossFunction(output, labelBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            testOutput = recognizer(testImage)
            predictLabel = torch.max(testOutput, 1)[1].data.numpy()
            accuracy = float((predictLabel == testLabel).astype(int).sum())/float(testLabel.size)
            print 'EPOCH: ', epoch, '\t|train loss: %.4f' % loss.data.numpy(), '\ttest accuracy: %.2f' % accuracy


testOutput = recognizer(testImage[:10].view(-1, 28, 28))
predictLabel = torch.max(testOutput, 1)[1].data.numpy()
print(predictLabel, 'predicted number')
print(testLabel[:10], 'real nukber')