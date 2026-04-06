import random

from ..action import Action
from ..agent import Agent
from ..environment import State


class RandomAgent(Agent):
    def __init__(self, probability_draw: float = 0.5):
        self.probability_draw = probability_draw

    def step(self, state: State) -> Action:
        if random.random() < self.probability_draw:
            return Action.DRAW
        else:
            return Action.PASS
