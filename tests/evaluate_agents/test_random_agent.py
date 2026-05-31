import numpy as np
import pytest
from tqdm import tqdm

from src.environment import Flip7Env
from src.main import run_episode
from src.simple_agents.random_agent import RandomAgent
from tests.evaluate_agents.utils import NUMBER_OF_ITERATIONS, store_results


@pytest.fixture
def flip7_env() -> Flip7Env:
    return Flip7Env()


@pytest.mark.parametrize("probability_draw", [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test_random_agent(flip7_env, probability_draw):
    agent = RandomAgent(probability_draw=probability_draw)

    scores = []
    for _ in tqdm(range(NUMBER_OF_ITERATIONS)):
        score = run_episode(agent, flip7_env)
        scores.append(score)

    # Store results
    store_results(
        scores,
        agent_name=agent.__class__.__name__,
        mark=f"probability_draw_{probability_draw}",
        agent_params={"probability_draw": probability_draw},
    )
    assert 0 <= np.mean(scores) <= 100
