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


@pytest.mark.parametrize(
    "alpha, gamma, temperature, penalty_for_passing, learning_rounds",
    [
        (0.5, 0.8, 1.0, 12, int(2e4)),
    ],
)
def test_random_agent(flip7_env, alpha, gamma, temperature, penalty_for_passing, learning_rounds):
    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        temperature=temperature,
        penalty_for_passing=penalty_for_passing,
    )

    for _ in tqdm(range(int(learning_rounds))):
        run_episode(agent, flip7_env, train=True)

    agent.temperature = 0.5

    scores = []
    for _ in tqdm(range(NUMBER_OF_ITERATIONS)):
        score = run_episode(agent, flip7_env)
        scores.append(score)

    # Store results
    store_results(
        scores,
        agent_name=agent.__class__.__name__,
        mark=f"{alpha}_{gamma}_{temperature}_{penalty_for_passing}_{learning_rounds}",
        agent_params={
            "alpha": alpha,
            "gamma": gamma,
            "temperature": temperature,
            "penalty_for_passing": penalty_for_passing,
            "learning_rounds": learning_rounds,
        },
    )
    assert 0 <= np.mean(scores) <= 100
