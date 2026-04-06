from ..action import Action
from ..agent import Agent
from ..environment import State


class NumberOfCardsThresholdAgent(Agent):
    def __init__(self, number_of_cards_threshold: int):
        self.number_of_cards_threshold = number_of_cards_threshold

    def step(self, state: State) -> Action:
        number_of_cards = state.hand.get_number_of_cards()
        if number_of_cards >= self.number_of_cards_threshold:
            return Action.PASS
        else:
            return Action.DRAW
