from types import SimpleNamespace

import pytest

from meridian_agent import config as C
from meridian_agent.data import Dataset
from meridian_agent.tools import RunState


@pytest.fixture(scope="session")
def ds():
    return Dataset(C.ROOT / "data")


@pytest.fixture()
def state(ds):
    return RunState.new(ds)


class FakeClient:
    """Scripted stand-in for the Anthropic client so the agent loop can be tested without an API key."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if not self.script:
            raise AssertionError("agent asked for more model turns than the script provides")
        return self.script.pop(0)


_n = [0]


def tool(tool_name, /, **inp):
    _n[0] += 1
    return SimpleNamespace(type="tool_use", id=f"toolu_{_n[0]}", name=tool_name, input=inp)


def text(t):
    return SimpleNamespace(type="text", text=t)


def resp(*blocks):
    return SimpleNamespace(content=list(blocks), stop_reason="tool_use")
