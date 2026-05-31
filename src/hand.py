from typing import Self

from .card import UNIQUE_CARDS, Card


class Hand:
    SEVEN_CARD_BONUS = 15

    def __init__(self):
        self.cards: dict[Card, int] = {card: 0 for card in UNIQUE_CARDS}

    def add_card(self, card: Card):
        self.cards[card] += 1

    def get_number_of_cards(self) -> int:
        return sum(self.cards.values())

    def is_bust(self) -> bool:
        # A hand is bust if it contains any card more than once
        return max(self.cards.values()) > 1

    def is_flip_seven(self) -> bool:
        # A hand is a flip seven if it contains exactly 7 cards and all are unique
        return self.get_number_of_cards() == 7 and not self.is_bust()

    def get_score(self) -> int:
        if self.is_bust():
            return 0

        score = sum([card.score * amount for card, amount in self.cards.items()])

        if self.is_flip_seven():
            score += self.SEVEN_CARD_BONUS

        return score

    def copy(self) -> Self:
        new_hand = Hand()
        new_hand.cards = self.cards.copy()
        return new_hand

    def get_representation(self) -> list[int]:
        # Return a list of counts of each card in the hand, in the same order as UNIQUE_CARDS
        return [self.cards[card] for card in UNIQUE_CARDS]
