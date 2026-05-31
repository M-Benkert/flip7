import numpy as np
import pytest
from tqdm import tqdm

from src.environment import Flip7Env
from src.main import run_episode
from src.q_learning.q_learning_agent import QLearningAgent
from tests.evaluate_agents.utils import NUMBER_OF_ITERATIONS, store_results


@pytest.fixture
def flip7_env() -> Flip7Env:
    return Flip7Env()


@pytest.mark.parametrize("learning_rounds", [1e3, 1e4, 1e5, 1e6])
def test_random_agent(flip7_env, learning_rounds):
    agent = QLearningAgent()

    for _ in tqdm(range(int(learning_rounds))):
        run_episode(agent, flip7_env, train=True)

    scores = []
    for _ in tqdm(range(NUMBER_OF_ITERATIONS)):
        score = run_episode(agent, flip7_env)
        scores.append(score)

    # Store results
    store_results(
        scores,
        agent_name=agent.__class__.__name__,
        mark=f"learning_rounds_{learning_rounds}",
        agent_params={"learning_rounds": learning_rounds},
    )
    assert 0 <= np.mean(scores) <= 100
