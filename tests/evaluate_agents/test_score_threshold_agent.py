import numpy as np
import pytest
from tqdm import tqdm

from src.environment import Flip7Env
from src.main import run_episode
from src.simple_agents.score_threshold_agent import ScoreThresholdAgent
from tests.evaluate_agents.utils import NUMBER_OF_ITERATIONS, store_results


@pytest.fixture
def flip7_env() -> Flip7Env:
    return Flip7Env()


@pytest.mark.parametrize("score_threshold", [10, 15, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32, 35, 40, 50])
def test_score_threshold_agent(flip7_env, score_threshold):
    agent = ScoreThresholdAgent(score_threshold=score_threshold)

    scores = []
    for _ in tqdm(range(NUMBER_OF_ITERATIONS)):
        score = run_episode(agent, flip7_env)
        scores.append(score)

    # Store results
    store_results(
        scores,
        agent_name=agent.__class__.__name__,
        mark=f"score_threshold{score_threshold}",
        agent_params={"score_threshold": score_threshold},
    )
    assert 0 <= np.mean(scores) <= 100
