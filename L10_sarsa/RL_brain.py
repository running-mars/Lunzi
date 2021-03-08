import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, actionSpace, learningRate=0.01, rewardDecay=0.9, eGreedy=0.9):
        self.actions = actionSpace
        self.learningRate = learningRate
        self.gamma = rewardDecay
        self.epsilon = eGreedy
        self.qTable = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.qTable.index:
            self.qTable = self.qTable.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.qTable.columns,
                    name=state
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:
            stateAction = self.qTable.loc[observation, :]
            action = np.random.choice(stateAction[stateAction == np.max(stateAction)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


class SarsaTable(RL):
    def __init__(self, actions, learningRate=0.01, rewardDecay=0.9, eGreedy=0.9):
        super(SarsaTable, self).__init__(actions, learningRate, rewardDecay, eGreedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        qPredict = self.qTable.loc[s, a]
        if s_ != 'terminal':
            qTarget = r + self.gamma*self.qTable.loc[s_, a_]
        else:
            qTarget = r
        self.qTable.loc[s, a] += self.learningRate*(qTarget - qPredict)


class QLearningTable(RL):
    def __init__(self, actions, learningRate=0.01, rewardDecay=0.9, eGreedy=0.9):
        super(QLearningTable, self).__init__(actions, learningRate, rewardDecay, eGreedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        qPredict = self.qTable.loc[s, a]
        if s_ != 'terminal':
            qTarget = r + self.gamma*self.qTable[s_, :].max()
        else:
            qTarget = r
        self.qTable.loc[s, a] += self.learningRate*(qTarget - qPredict)