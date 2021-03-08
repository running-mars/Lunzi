import numpy as np 
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk 
else:
    import tkinter as tk 

UNIT = 40   # pixels per cm
MAZE_H = 4
MAZE_W = 4


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.actionSpace = ['u', 'd', 'l', 'r']
        self.actionNumber = len(self.actionSpace)
        self.title('Maze')
        self.geometry('{0}x{1}'.format(MAZE_H*UNIT, MAZE_W*UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H*UNIT, width=MAZE_W*UNIT)
        # create grids
        for column in range(0, MAZE_W*UNIT, UNIT):
            x0, y0, x1, y1 = column, 0, column, MAZE_H*UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for row in range(0, MAZE_H*UNIT, UNIT):
            x0, y0, x1, y1 = 0, row, MAZE_W*UNIT, row
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell 1
        hell1Center = origin + np.array([UNIT*2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1Center[0] - 15, hell1Center[1] - 15,
            hell1Center[0] + 15, hell1Center[1] + 15,
            fill='black')
        # hell 2
        hell2Center = origin + np.array([UNIT, UNIT*2])
        self.hell2 = self.canvas.create_rectangle(
            hell2Center[0] - 15, hell2Center[1] - 15,
            hell2Center[0] + 15, hell2Center[1] + 15,
            fill='black')
        # oval
        ovalCenter = origin + UNIT*2
        self.oval = self.canvas.create_oval(
            ovalCenter[0] - 15, ovalCenter[1] - 15,
            ovalCenter[0] + 15, ovalCenter[1] + 15,
            fill='yellow'
        )
        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        state = self.canvas.coords(self.rect)
        baseAction = np.array([0, 0])
        if action == 0:
            if state[1] > UNIT:
                baseAction[1] -= UNIT
        elif action == 1:
            if state[1] < (MAZE_H - 1)*UNIT:
                baseAction[1] += UNIT
        elif action == 2:
            if state[0] < (MAZE_W - 1)*UNIT:
                baseAction[0] += UNIT
        elif action == 3:
            if state[0] > UNIT:
                baseAction[0] -= UNIT
        self.canvas.move(self.rect, baseAction[0], baseAction[1])
        self.render()
        s_ = self.canvas.coords(self.rect)
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()
