"""Value-object semantics for every concrete :class:`ToolCall` subclass."""

from __future__ import annotations

import dataclasses
import pickle

import pytest

from rath.backend import (
    CodeRun,
    CommandRun,
    FilesExists,
    FilesList,
    FilesRead,
    FilesWrite,
    ToolCall,
)

_HASHABLE_CALLS = [
    CommandRun(cmd=("ls",)),
    CommandRun(cmd="ls -lah", timeout=5.0),
    FilesRead(path="/etc/hostname"),
    FilesRead(path="/x", encoding=None),
    FilesWrite(path="/x", data=b"abc"),
    FilesWrite(path="/x", data="abc", mode=0o600),
    FilesList(path="/"),
    FilesExists(path="/"),
    CodeRun(code="print(1)"),
    CodeRun(code="print(1)", language="python", timeout=2.0),
]


@pytest.mark.parametrize("call", _HASHABLE_CALLS)
def test_call_is_subclass_of_tool_call(call: ToolCall) -> None:
    assert isinstance(call, ToolCall)


@pytest.mark.parametrize("call", _HASHABLE_CALLS)
def test_call_is_frozen(call: ToolCall) -> None:
    field_name = dataclasses.fields(call)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(call, field_name, "should-fail")


@pytest.mark.parametrize("call", _HASHABLE_CALLS)
def test_call_pickle_round_trip(call: ToolCall) -> None:
    revived = pickle.loads(pickle.dumps(call))
    assert revived == call
    assert revived is not call


@pytest.mark.parametrize("call", _HASHABLE_CALLS)
def test_call_hash_stable(call: ToolCall) -> None:
    assert hash(call) == hash(call)


@pytest.mark.parametrize("call", _HASHABLE_CALLS)
def test_call_equals_self(call: ToolCall) -> None:
    assert call == call


def test_distinct_calls_inequal() -> None:
    assert CommandRun(cmd=("a",)) != CommandRun(cmd=("b",))
    assert FilesRead(path="/a") != FilesRead(path="/b")
    assert FilesRead(path="/a") != FilesExists(path="/a")


def test_command_run_defaults() -> None:
    c = CommandRun(cmd="ls")
    assert c.env is None
    assert c.cwd is None
    assert c.stdin is None
    assert c.timeout is None


def test_files_read_text_default_encoding() -> None:
    assert FilesRead(path="/x").encoding == "utf-8"


def test_files_write_default_mode() -> None:
    assert FilesWrite(path="/x", data=b"").mode == 0o644


def test_code_run_default_language_is_python() -> None:
    assert CodeRun(code="").language == "python"


def test_calls_have_slots() -> None:
    """frozen + slots is what gives us the value-object guarantee.

    With ``slots=True`` adding a new attribute raises ``AttributeError`` even
    via :func:`object.__setattr__` (which bypasses the frozen check).
    """
    c = CommandRun(cmd="ls")
    with pytest.raises(AttributeError):
        object.__setattr__(c, "brand_new_attr", "nope")


def test_calls_with_unhashable_field_remain_equal() -> None:
    """A dict env is unhashable but ``==`` still works as expected."""
    a = CommandRun(cmd="ls", env={"X": "1"})
    b = CommandRun(cmd="ls", env={"X": "1"})
    assert a == b
