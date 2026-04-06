from ..action import Action
from ..agent import Agent
from ..environment import State


class ScoreThresholdAgent(Agent):
    def __init__(self, score_threshold: int):
        self.score_threshold = score_threshold

    def step(self, state: State) -> Action:
        score = state.hand.get_score()
        if score >= self.score_threshold:
            return Action.PASS
        else:
            return Action.DRAW
