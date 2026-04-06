from ..action import Action
from ..agent import Agent
from ..environment import State


class CombinedThresholdAgent(Agent):
    def __init__(self, score_threshold: int, number_of_cards_threshold: int):
        self.score_threshold = score_threshold
        self.number_of_cards_threshold = number_of_cards_threshold

    def step(self, state: State) -> Action:
        score = state.hand.get_score()
        number_of_cards = state.hand.get_number_of_cards()

        if score >= self.score_threshold or number_of_cards >= self.number_of_cards_threshold:
            return Action.PASS
        else:
            return Action.DRAW
