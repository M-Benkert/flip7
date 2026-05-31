import numpy as np
import pytest
from tqdm import tqdm

from src.environment import Flip7Env
from src.main import run_episode
from src.simple_agents.number_of_cards_threshold_agent import NumberOfCardsThresholdAgent
from tests.evaluate_agents.utils import NUMBER_OF_ITERATIONS, store_results


@pytest.fixture
def flip7_env() -> Flip7Env:
    return Flip7Env()


@pytest.mark.parametrize("number_of_cards_threshold", list(range(1, 8)))
def test_number_of_cards_threshold_agent(flip7_env, number_of_cards_threshold):
    agent = NumberOfCardsThresholdAgent(number_of_cards_threshold=number_of_cards_threshold)

    scores = []
    for _ in tqdm(range(NUMBER_OF_ITERATIONS)):
        score = run_episode(agent, flip7_env)
        scores.append(score)

    # Store results
    store_results(
        scores,
        agent_name=agent.__class__.__name__,
        mark=f"number_of_cards_threshold{number_of_cards_threshold}",
        agent_params={"number_of_cards_threshold": number_of_cards_threshold},
    )
    assert 0 <= np.mean(scores) <= 100
