import numpy as np

from .card import Card, create_full_deck


class NoMoreCards(StopIteration):
    pass


class CardDeck:
    def __init__(self):
        self.cards: list[Card] = []
        self.reset()
        self.shuffle()

    def reset(self):
        self.cards = create_full_deck()

    def shuffle(self):
        self.cards = np.random.permutation(self.cards).tolist()

    def is_empty(self) -> bool:
        return len(self.cards) == 0

    def draw(self) -> Card:
        if self.is_empty():
            raise NoMoreCards("No more cards in the deck")
        return self.cards.pop()
