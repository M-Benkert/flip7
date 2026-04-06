from .agent import Agent
from .environment import Flip7Env


def run_episode(agent: Agent, env: Flip7Env | None = None) -> int:
    if env is None:
        env = Flip7Env()

    env.reset()
    done = False

    current_state = env.get_state()

    while not done:
        action = agent.step(current_state)
        done, current_state = env.step(action)

    return env.get_state().hand.get_score()
