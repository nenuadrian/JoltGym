"""Tests for the HalfCheetah environment."""

import numpy as np
import pytest


def test_env_creates():
    """Environment can be created and has correct spaces."""
    import joltgym
    env = joltgym.make("JoltGym/HalfCheetah-v0")
    assert env.observation_space.shape == (17,)
    assert env.action_space.shape == (6,)
    env.close()


def test_reset():
    """Reset returns valid observation."""
    import joltgym
    env = joltgym.make("JoltGym/HalfCheetah-v0")
    obs, info = env.reset(seed=42)
    assert obs.shape == (17,)
    assert not np.any(np.isnan(obs))
    env.close()


def test_step():
    """Step returns valid (obs, reward, term, trunc, info)."""
    import joltgym
    env = joltgym.make("JoltGym/HalfCheetah-v0")
    obs, _ = env.reset(seed=42)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (17,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "x_position" in info
    assert "x_velocity" in info
    env.close()


def test_multiple_steps():
    """Run 100 steps without NaN or crash."""
    import joltgym
    env = joltgym.make("JoltGym/HalfCheetah-v0")
    obs, _ = env.reset(seed=42)

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert not np.any(np.isnan(obs)), "NaN in observation"
        assert np.isfinite(reward), "Non-finite reward"

    env.close()


def test_deterministic():
    """Same seed produces same trajectory."""
    import joltgym

    def run_episode(seed):
        env = joltgym.make("JoltGym/HalfCheetah-v0")
        obs, _ = env.reset(seed=seed)
        np.random.seed(seed)
        rewards = []
        for _ in range(50):
            action = np.random.uniform(-1, 1, 6).astype(np.float32)
            obs, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        env.close()
        return np.array(rewards)

    r1 = run_episode(123)
    r2 = run_episode(123)
    np.testing.assert_array_almost_equal(r1, r2, decimal=4)
