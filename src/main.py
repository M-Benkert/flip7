from .agent import Agent
from .environment import Flip7Env


def run_episode(agent: Agent, env: Flip7Env | None = None, train: bool = False) -> int:
    if env is None:
        env = Flip7Env()

    env.reset()
    done = False

    next_state = env.get_state()

    while not done:
        state = next_state
        action = agent.step(state)
        done, next_state = env.step(action)
        if train:
            agent.learn(state, action, next_state, done)

    return env.get_state().hand.get_score()
