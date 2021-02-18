"""
batch training
"""
import torch
import torch.utils.data as Data

torch.manual_seed(1)    # make the result reproducible
BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)


def showBatch():
    for epoch in range(3):
        for step, (xBatch, yBatch) in enumerate(loader):
            # training process
            print 'Epoch: ', epoch, '\t| Step: ', step, '\t| xBatch: ', xBatch.numpy(), '\t| yBatch: ', yBatch.numpy()


if __name__ == '__main__':
    showBatch()
