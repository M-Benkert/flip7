import json
from typing import Self, TypeAlias

import numpy as np

from ..action import POSSIBLE_ACTIONS, Action
from ..agent import Agent
from ..environment import State

# Define a type alias for the possible states in the environment
# A tuple representing the count of each card in the hand, in the same order as UNIQUE_CARDS
StateType: TypeAlias = tuple[int, ...]


def parse_state(state: State) -> StateType:
    return tuple(state.hand.get_representation())


class QLearningAgent(Agent):
    def __init__(self, alpha=0.5, gamma=0.8, temperature=1.0, penalty_for_passing=12):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.temperature = temperature  # Temperature for softmax action selection

        self.penalty_for_passing = penalty_for_passing

        self.q_table: dict[tuple[StateType, Action], float] = {}  # Q-table to store Q-values

    def get_q_value(self, state: StateType, action: Action) -> float:
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state: StateType) -> Action:
        # use softmax strategy for exploration instead of epsilon-greedy
        q_values = [self.get_q_value(state, a) for a in POSSIBLE_ACTIONS]
        max_q = max(q_values)

        exp_values = [np.exp((q - max_q) / self.temperature) for q in q_values]
        sum_exp = sum(exp_values)

        probabilities = [exp_q / sum_exp for exp_q in exp_values]

        return np.random.choice(POSSIBLE_ACTIONS, p=probabilities)

    def calculate_reward(self, state: State, action: Action, next_state: State, done: bool) -> float:
        if action == Action.PASS:
            return next_state.hand.get_score() / 3 - self.penalty_for_passing

        reward = next_state.hand.get_score() - state.hand.get_score()

        return reward / 2

    def update_q_value(self, state: StateType, action: Action, reward: float, next_state: StateType, done: bool):
        max_future_q = 0 if done else max(self.get_q_value(next_state, a) for a in POSSIBLE_ACTIONS)

        self.q_table[(state, action)] = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (
            reward + self.gamma * max_future_q
        )

    def step(self, state: State) -> Action:
        action = self.choose_action(parse_state(state))
        return action

    def learn(self, state: State, action: Action, next_state: State, done: bool):
        reward = self.calculate_reward(state, action, next_state, done)
        self.update_q_value(parse_state(state), action, reward, parse_state(next_state), done)

    def serialize(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "temperature": self.temperature,
            "q_table": {json.dumps((key[0], key[1].value)): value for key, value in self.q_table.items()},
        }

    @classmethod
    def deserialize(cls, data: dict) -> Self:
        agent = cls(
            alpha=data["alpha"],
            gamma=data["gamma"],
            temperature=data["temperature"],
        )
        agent.q_table = {
            (tuple(json.loads(key)[0]), Action(json.loads(key)[1])): value for key, value in data["q_table"].items()
        }
        return agent
