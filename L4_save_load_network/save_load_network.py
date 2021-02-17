"""
save and load, network and parameters
"""
import torch
import matplotlib.pyplot as plt

# make some data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())


def trainingFitNet():
    fitter = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(fitter.parameters(), lr=0.5)
    lossFunction = torch.nn.MSELoss()

    for t in range(100):
        prediction = fitter(x)
        loss = lossFunction(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title("fitter")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    torch.save(fitter, 'fitter.pkl')
    torch.save(fitter.state_dict(), 'fitter_param.pkl')


def restoreFitter():
    fitterRestored = torch.load("fitter.pkl")
    prediction = fitterRestored(x)

    plt.subplot(132)
    plt.title("fitter restored")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def restoreParameter():
    fitter = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    fitter.load_state_dict(torch.load("fitter_param.pkl"))
    prediction = fitter(x)

    plt.subplot(133)
    plt.title("fitter parameters restored")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

if __name__ == "__main__":
    trainingFitNet()
    restoreFitter()
    restoreParameter()
    plt.show()
