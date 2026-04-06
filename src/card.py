from dataclasses import dataclass

from .card_value import CardValue


@dataclass
class Card:
    value: CardValue
    score: int

    # override hash function
    def __hash__(self):
        return hash(self.value)


# Unique cards - one instance of each card value
UNIQUE_CARDS = [
    Card(CardValue.ZERO, 0),
    Card(CardValue.ONE, 1),
    Card(CardValue.TWO, 2),
    Card(CardValue.THREE, 3),
    Card(CardValue.FOUR, 4),
    Card(CardValue.FIVE, 5),
    Card(CardValue.SIX, 6),
    Card(CardValue.SEVEN, 7),
    Card(CardValue.EIGHT, 8),
    Card(CardValue.NINE, 9),
    Card(CardValue.TEN, 10),
    Card(CardValue.ELEVEN, 11),
    Card(CardValue.TWELVE, 12),
]

# Define how many copies of each card are in the deck
CARD_COUNTS: dict[CardValue, int] = {
    CardValue.ZERO: 1,
    CardValue.ONE: 1,
    CardValue.TWO: 2,
    CardValue.THREE: 3,
    CardValue.FOUR: 4,
    CardValue.FIVE: 5,
    CardValue.SIX: 6,
    CardValue.SEVEN: 7,
    CardValue.EIGHT: 8,
    CardValue.NINE: 9,
    CardValue.TEN: 10,
    CardValue.ELEVEN: 11,
    CardValue.TWELVE: 12,
}


def create_full_deck() -> list[Card]:
    deck = []
    for card in UNIQUE_CARDS:
        count = CARD_COUNTS.get(card.value, 0)
        for _ in range(count):
            deck.append(Card(card.value, card.score))
    return deck
