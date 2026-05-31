from enum import Enum


class Action(Enum):
    PASS = 0
    DRAW = 1


POSSIBLE_ACTIONS = [Action.DRAW, Action.PASS]
