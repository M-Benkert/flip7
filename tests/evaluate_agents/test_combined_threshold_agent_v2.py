from itertools import product

import numpy as np
import pytest
from tqdm import tqdm

from src.environment import Flip7Env
from src.main import run_episode
from src.simple_agents.combined_threshold_agent_v2 import CombinedThresholdAgentV2
from tests.evaluate_agents.utils import NUMBER_OF_ITERATIONS, store_results


@pytest.fixture
def flip7_env() -> Flip7Env:
    return Flip7Env()


@pytest.mark.parametrize(
    "score_threshold, number_of_cards_threshold", list(product([20, 22, 24, 26, 28, 30], [2, 3, 4]))
)
def test_combined_threshold_agent_v2(flip7_env, score_threshold, number_of_cards_threshold):
    agent = CombinedThresholdAgentV2(
        score_threshold=score_threshold, number_of_cards_threshold=number_of_cards_threshold
    )

    scores = []
    for _ in tqdm(range(NUMBER_OF_ITERATIONS)):
        score = run_episode(agent, flip7_env)
        scores.append(score)

    # Store results
    store_results(
        scores,
        agent_name=agent.__class__.__name__,
        mark=f"combined_{score_threshold}_{number_of_cards_threshold}",
        agent_params={"score_threshold": score_threshold, "number_of_cards_threshold": number_of_cards_threshold},
    )
    assert 0 <= np.mean(scores) <= 100
