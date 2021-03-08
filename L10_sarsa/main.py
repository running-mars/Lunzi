from maze_env import Maze
from RL_brain import SarsaTable, QLearningTable


def update():
    for episode in range(1000):
        observation = env.reset()
        action = RL.choose_action(str(observation))
        while True:
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            # sarsa
            # RL.learn(str(observation), action, reward, str(observation_), action_)
            # QLeaning
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            action = action_

            if done:
                break
        print("Episode: %d | reward: %f" % (episode, reward))
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.actionNumber)))
    env.after(100, update)
    env.mainloop()
