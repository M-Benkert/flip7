import json

import pytest

from src.action import Action
from src.q_learning.q_learning_agent import QLearningAgent


class TestQLearningAgent:
    def test_init(self):
        agent = QLearningAgent(alpha=0.1, gamma=0.9, temperature=1.0)
        assert agent.alpha == 0.1
        assert agent.gamma == 0.9
        assert agent.temperature == 1.0
        assert isinstance(agent.q_table, dict)

    @pytest.mark.parametrize("action", [Action.PASS, Action.DRAW])
    def test_get_q_value(self, action):
        agent = QLearningAgent()
        state = (0, 0, 0, 0, 0, 0, 0)  # Example state
        assert agent.get_q_value(state, action) == 0.0

    @pytest.mark.parametrize(
        "state, action", [((0, 0, 0, 0, 0, 0, 0), Action.PASS), ((1, 0, 0, 0, 0, 0, 0), Action.DRAW)]
    )
    def test_get_q_value_with_q_table(self, state, action):
        agent = QLearningAgent()
        agent.q_table[(state, action)] = 5.0
        assert agent.get_q_value(state, action) == 5.0

    def test_update_q_value(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9)
        state = (0, 0, 0, 0, 0, 0, 0)
        action = Action.DRAW
        reward = 10
        next_state = (1, 0, 0, 0, 0, 0, 0)
        done = False

        agent.update_q_value(state, action, reward, next_state, done)

        expected_q_value = 5.0  # Since initial Q-value is 0 and max_future_q is also 0
        assert agent.get_q_value(state, action) == expected_q_value

    def test_update_q_value_with_future_q(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9)
        state = (0, 0, 0, 0, 0, 0, 0)
        action = Action.PASS
        reward = 10
        next_state = (1, 0, 0, 0, 0, 0, 0)
        done = False

        # Set a future Q-value for the next state and an action
        agent.q_table[(next_state, Action.DRAW)] = 4.0

        agent.update_q_value(state, action, reward, next_state, done)

        expected_q_value = (1 - agent.alpha) * 0 + agent.alpha * (reward + agent.gamma * 4.0)
        assert agent.get_q_value(state, action) == expected_q_value

    def test_update_q_value__with_future_q_and_done(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9)
        state = (0, 0, 0, 0, 0, 0, 0)
        action = Action.PASS
        reward = 10
        next_state = (1, 0, 0, 0, 0, 0, 0)
        done = True

        # Set a future Q-value for the next state and an action
        agent.q_table[(next_state, Action.DRAW)] = 4.0

        agent.update_q_value(state, action, reward, next_state, done)

        expected_q_value = (1 - agent.alpha) * 0 + agent.alpha * reward
        assert agent.get_q_value(state, action) == expected_q_value

    def test_serialize(self):
        alpha = 0.1
        gamma = 0.9
        temperature = 1.0

        agent = QLearningAgent(alpha=alpha, gamma=gamma, temperature=temperature)

        serialized = agent.serialize()

        assert serialized["alpha"] == alpha
        assert serialized["gamma"] == gamma
        assert serialized["temperature"] == temperature

    def test_serialize_with_q_table(self):
        state = (0, 0, 0, 0, 0, 0, 0)
        action = Action.PASS
        q_value = 5.0

        agent = QLearningAgent()
        agent.q_table[(state, action)] = q_value
        serialized = agent.serialize()

        expected_q_table = {json.dumps((state, action.value)): q_value}
        assert serialized["q_table"] == expected_q_table

    def test_deserialize(self):
        alpha = 0.1
        gamma = 0.9
        temperature = 1.0

        data = {
            "alpha": alpha,
            "gamma": gamma,
            "temperature": temperature,
            "q_table": {},
        }

        agent = QLearningAgent.deserialize(data)

        assert agent.alpha == alpha
        assert agent.gamma == gamma
        assert agent.temperature == temperature
        assert isinstance(agent.q_table, dict)

    def test_deserialize_with_q_table(self):
        state = (0, 0, 0, 0, 0, 0, 0)
        action = Action.PASS
        q_value = 5.0

        data = {
            "alpha": 0.1,
            "gamma": 0.9,
            "temperature": 1.0,
            "q_table": {json.dumps((state, action.value)): q_value},
        }

        agent = QLearningAgent.deserialize(data)

        assert agent.get_q_value(state, action) == q_value

    def test_serialize_deserialize_cycle(self):
        agent = QLearningAgent(alpha=0.1, gamma=0.9, temperature=1.0)
        state = (0, 0, 0, 0, 0, 0, 0)
        action = Action.DRAW
        q_value = 5.0
        agent.q_table[(state, action)] = q_value

        serialized = agent.serialize()
        deserialized_agent = QLearningAgent.deserialize(serialized)

        assert deserialized_agent.alpha == agent.alpha
        assert deserialized_agent.gamma == agent.gamma
        assert deserialized_agent.temperature == agent.temperature
        assert deserialized_agent.get_q_value(state, action) == q_value
