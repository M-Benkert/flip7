import pytest

from src.card import UNIQUE_CARDS, Card
from src.hand import Hand


@pytest.fixture
def card_ten() -> Card:
    return UNIQUE_CARDS[10]  # Card with score 10


@pytest.fixture
def card_twelve() -> Card:
    return UNIQUE_CARDS[12]  # Card with score 12


@pytest.fixture
def hand_instance() -> Hand:
    return Hand()


class TestHand:
    def test_init(self):
        hand = Hand()
        for card in UNIQUE_CARDS:
            assert hand.cards[card] == 0

    def test_add_card(self, hand_instance, card_ten):
        hand_instance.add_card(card_ten)
        assert hand_instance.cards[card_ten] == 1

        hand_instance.add_card(card_ten)
        assert hand_instance.cards[card_ten] == 2

    def test_get_number_of_cards(self, hand_instance, card_ten, card_twelve):
        assert hand_instance.get_number_of_cards() == 0

        hand_instance.add_card(card_ten)
        assert hand_instance.get_number_of_cards() == 1

        hand_instance.add_card(card_ten)
        assert hand_instance.get_number_of_cards() == 2

        hand_instance.add_card(card_twelve)
        assert hand_instance.get_number_of_cards() == 3

    def test_is_bust(self, hand_instance, card_ten, card_twelve):
        # Not bust initially
        assert not hand_instance.is_bust()

        # Add unique cards
        hand_instance.add_card(card_ten)
        hand_instance.add_card(card_twelve)
        assert not hand_instance.is_bust()

        # Add duplicate
        hand_instance.add_card(card_ten)
        assert hand_instance.is_bust()

    @pytest.mark.parametrize(
        "cards, expected",
        [
            ([], False),  # No cards
            ([UNIQUE_CARDS[0]], False),  # One unique card
            ([UNIQUE_CARDS[i] for i in range(2)], False),  # Two unique cards
            ([UNIQUE_CARDS[i] for i in range(3)], False),  # Three unique cards
            ([UNIQUE_CARDS[i] for i in range(4)], False),  # Four unique cards
            ([UNIQUE_CARDS[i] for i in range(4)], False),  # Five unique cards
            ([UNIQUE_CARDS[i] for i in range(6)], False),  # Siv unique cards
            ([UNIQUE_CARDS[i] for i in range(7)], True),  # Seven unique cards (flip seven)
            ([UNIQUE_CARDS[i] for i in range(6)] + [UNIQUE_CARDS[0]], False),  # Six unique + duplicate
        ],
    )
    def test_is_flip_seven(self, hand_instance, cards, expected):
        for card in cards:
            hand_instance.add_card(card)
        assert hand_instance.is_flip_seven() == expected

    def test_get_score_simple(self, hand_instance, card_ten, card_twelve):
        assert hand_instance.get_score() == 0  # No cards

        hand_instance.add_card(card_ten)
        assert hand_instance.get_score() == 10  # One card

        hand_instance.add_card(card_twelve)
        assert hand_instance.get_score() == 22  # Two cards

    def test_get_score_bust(self, hand_instance, card_ten):
        hand_instance.add_card(card_ten)
        hand_instance.add_card(card_ten)  # Duplicate card, should be bust
        assert hand_instance.get_score() == 0  # Bust hand should have score 0

    def test_get_score_flip_seven(self, hand_instance):
        for i in range(7):
            hand_instance.add_card(UNIQUE_CARDS[i])  # Add seven unique cards
        expected_score = (
            sum(UNIQUE_CARDS[i].score for i in range(7)) + Hand.SEVEN_CARD_BONUS
        )  # Base score + flip seven bonus
        assert hand_instance.get_score() == expected_score
