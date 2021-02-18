"""
pyglet version: 1.2.4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.output = nn.Linear(50, NUM_ACTIONS)
        self.output.weight.data.normal_(0, 0.1)

    def forward(self, inputFeature):
        featureAfterFc1 = self.fc1(inputFeature)
        featureAfterFc1 = F.relu(featureAfterFc1)
        action = self.output(featureAfterFc1)
        return action


class DQN(object):
    def __init__(self):
        self.evalNet = Net()
        self.targetNet = Net()

        self.learnStepCounter = 0
        self.memoryCounter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES*2 + 2))
        self.optimizer = torch.optim.Adam(self.evalNet.parameters(), lr=LR)
        self.lossFunction = nn.MSELoss()

    def chooseAction(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.uniform() < EPSILON:
            action = self.evalNet.forward(state)
            action = torch.max(action, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0, NUM_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def storeTransition(self, state, action, reward, nextState):
        transition = np.hstack((state, [action, reward], nextState))
        index = self.memoryCounter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memoryCounter += 1

    def learn(self):
        if self.learnStepCounter % TARGET_REPLACE_ITER == 0:
            self.targetNet.load_state_dict(self.evalNet.state_dict())
        self.learnStepCounter += 1

        # sample batch transitions
        sampleIndex = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        memoryBatch = self.memory[sampleIndex, :]
        stateBatch = torch.FloatTensor(memoryBatch[:, :NUM_STATES])
        actionBatch = torch.LongTensor(memoryBatch[:, NUM_STATES:NUM_STATES+1].astype(int))
        rewardBatch = torch.FloatTensor(memoryBatch[:, NUM_STATES+1:NUM_STATES+2])
        nextStateBatch = torch.FloatTensor(memoryBatch[:, -NUM_STATES:])

        qEval = self.evalNet(stateBatch).gather(1, actionBatch)
        qNext = self.targetNet(nextStateBatch).detach()
        qTarget = rewardBatch + GAMMA*qNext.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.lossFunction(qEval, qTarget)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


controller = DQN()
print "\nCollecting experience...."
for numEpisode in range(400):
    state = env.reset()
    rewardEpisode = 0
    while True:
        env.render()
        action = controller.chooseAction(state)

        # take actions
        nextState, reward, done, info = env.step(action)
        x, dX, theta, dTheta = nextState
        reward1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        reward2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = reward1 + reward2

        controller.storeTransition(state, action, reward, nextState)

        rewardEpisode += reward

        if controller.memoryCounter > MEMORY_CAPACITY:
            controller.learn()
            if done:
                print "Episode: ", numEpisode, "\tEpisode Reward: ", round(rewardEpisode, 2)

        if done:
            break
        state = nextState





