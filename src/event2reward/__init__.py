"""Event2Reward: goal-conditioned task progress for robot observations."""
__version__ = "1.0.0"


def __getattr__(name):
    if name in {"Event2Reward", "Observation"}:
        from .inference import Event2Reward, Observation
        return {"Event2Reward": Event2Reward, "Observation": Observation}[name]
    raise AttributeError(name)
