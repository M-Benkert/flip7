from .action import Action
from .card import Card
from .card_deck import CardDeck
from .hand import Hand


class State:
    def __init__(self, hand: Hand):
        self.hand = hand


class Flip7Env:
    def __init__(self):
        self.card_deck = CardDeck()

        self.hand = Hand()
        self.score = 0
        self.done = False

        self.reset()

    def reset(self):
        self.card_deck.reset()
        self.card_deck.shuffle()

        self.hand = Hand()
        self.score = 0
        self.done = False

    def step(self, action: Action) -> tuple[bool, State]:
        if self.done:
            raise Exception("Episode is done. Please reset the environment.")

        if action == Action.PASS:
            self.done = True
        elif action == Action.DRAW:
            card = self._draw_card()
            self.hand.add_card(card)

            self.done = self.hand.is_bust() or self.hand.get_number_of_cards() >= 7

        return self.done, self.get_state()

    def _draw_card(self) -> Card:
        while self.card_deck.is_empty():
            self.card_deck.reset()
            self.card_deck.shuffle()
        return self.card_deck.draw()

    def get_state(self) -> State:
        return State(hand=self.hand)
