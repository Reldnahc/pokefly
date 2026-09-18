import importlib.util
from pathlib import Path


def load_probe():
    path = Path(__file__).parents[1] / "scripts" / "probe_operant_motor.py"
    spec = importlib.util.spec_from_file_location("operant_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedTestNeurons:
    """Unit fixture only, not a deployed controller."""

    def __init__(self):
        self.actions = ["up", "down", "up", "wait", "down", "up", "up", "up"]
        self.delivered = []

    def reset_dynamics(self, seed):
        self.index = 0

    def observe(self, frame):
        return None

    def choose(self, observation):
        action = self.actions[self.index]
        self.index += 1
        return action, {}

    def reinforce(self, reward, *, enabled):
        self.delivered.append((reward, enabled))


def test_delayed_operant_feedback_flushes_tail_without_rewarding_tail_actions():
    c = FixedTestNeurons()
    result = load_probe().episode(c, None, 1, 8, target="up", reward_delay=2, enabled=True)
    assert result["rewards"] == [0, 0, 1, 0, 1, 0, 0, 1]
    assert c.delivered == [(r, True) for r in result["rewards"]]
    assert result["actions"] == c.actions


def test_zero_delay_is_unchanged_and_yoked_feedback_is_not_delayed_twice():
    probe = load_probe()
    c = FixedTestNeurons()
    assert probe.episode(c, None, 1, 8, target="up")["rewards"] == [1, 0, 1, 0, 0, 1, 1, 1]
    schedule = [0, 1, 0, 0, 0, 1, 0, 1]
    c = FixedTestNeurons()
    result = probe.episode(c, None, 1, 8, schedule=schedule, reward_delay=2)
    assert result["rewards"] == schedule
    assert c.delivered == [(r, False) for r in schedule]
