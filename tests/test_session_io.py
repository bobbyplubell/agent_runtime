import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_runtime import load_session, save_session


def _sample_messages() -> list:
    tool_calls = [
        {
            "id": "call-123",
            "name": "workspace_exec",
            "type": "tool_call",
            "args": {"command": "echo hello"},
        }
    ]
    return [
        SystemMessage(content="system ready", name="system"),
        HumanMessage(content="hi", name="operator", additional_kwargs={"foo": "bar"}),
        AIMessage(
            content="running tool",
            name="assistant",
            tool_calls=tool_calls,
            response_metadata={"model": "unit-test"},
        ),
        ToolMessage(
            content=json.dumps({"ok": True}),
            name="workspace_exec",
            tool_call_id="call-123",
        ),
        AIMessage(content="done", name="assistant"),
    ]


def test_save_and_load_session_roundtrip(tmp_path: Path) -> None:
    session_path = tmp_path / "session.json"
    original = _sample_messages()

    metadata = {"workspace": {"workspace_id": "ws-1234"}}
    save_session(original, session_path, metadata)
    restored, restored_meta = load_session(session_path)

    assert len(restored) == len(original)
    assert restored_meta == metadata
    for before, after in zip(original, restored):
        assert before.type == after.type
        assert before.content == after.content
        assert getattr(after, "name", None) == getattr(before, "name", None)
        assert getattr(after, "additional_kwargs", None) == getattr(before, "additional_kwargs", None)
        assert getattr(after, "response_metadata", None) == getattr(before, "response_metadata", None)
        if hasattr(before, "tool_calls"):
            assert getattr(after, "tool_calls", None) == getattr(before, "tool_calls", None)
        if hasattr(before, "tool_call_id"):
            assert getattr(after, "tool_call_id", None) == getattr(before, "tool_call_id", None)
