from .action import Action
from .environment import State


class Agent:
    def step(self, state: State) -> Action:
        pass

    def learn(self, state: State, action: Action, next_state: State, done: bool):
        pass
