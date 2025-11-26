"""Agent runtime scaffolding shared across CLI, service, and orchestrator entrypoints."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import ValidationError

SUPPORTED_LLM_PROVIDERS = {"ollama", "openai", "deepseek", "openrouter"}

_DEFAULT_MODELS = {
    "ollama": "qwen3:30b-a3b",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-reasoner",
    "openrouter": "openrouter/auto",
}

_DEFAULT_BASE_URLS = {
    "ollama": None,
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
}

_GLOBAL_PROVIDER_ENV = ["AGENT_PROVIDER", "AGENT_LLM_PROVIDER"]
_GLOBAL_MODEL_ENV = ["AGENT_MODEL", "AGENT_LLM_MODEL"]
_GLOBAL_BASE_ENV = ["AGENT_BASE_URL", "AGENT_LLM_BASE_URL"]
_GLOBAL_TEMP_ENV = ["AGENT_TEMPERATURE", "AGENT_LLM_TEMPERATURE"]
_GLOBAL_API_KEY_ENV = ["AGENT_API_KEY", "AGENT_LLM_API_KEY"]

_MODEL_ENV_VARS = {
    "ollama": ["AGENT_OLLAMA_MODEL", "OLLAMA_MODEL"],
    "openai": ["AGENT_OPENAI_MODEL", "OPENAI_MODEL"],
    "deepseek": ["AGENT_DEEPSEEK_MODEL", "DEEPSEEK_MODEL"],
    "openrouter": ["AGENT_OPENROUTER_MODEL", "OPENROUTER_MODEL"],
}

_BASE_ENV_VARS = {
    "ollama": ["AGENT_OLLAMA_BASE_URL", "OLLAMA_BASE_URL"],
    "openai": ["AGENT_OPENAI_BASE_URL", "OPENAI_BASE_URL"],
    "deepseek": ["AGENT_DEEPSEEK_BASE_URL", "DEEPSEEK_BASE_URL"],
    "openrouter": ["AGENT_OPENROUTER_BASE_URL", "OPENROUTER_BASE_URL"],
}

_PROVIDER_API_KEY_ENV_VARS = {
    "openai": ["AGENT_OPENAI_API_KEY", "OPENAI_API_KEY"],
    "deepseek": [
        "AGENT_DEEPSEEK_API_KEY",
        "DEEPSEEK_API_KEY",
        "AGENT_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    ],
    "openrouter": ["AGENT_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"],
}

_SESSION_FILE_VERSION = 2


class ToolCallLogBuffer:
    """Track console output for each tool call so it can be removed later."""

    def __init__(self) -> None:
        self._call_logs: List[List[str]] = []

    def begin_call(self) -> None:
        self._call_logs.append([])

    def log_line(self, text: str) -> None:
        print(text)
        if self._call_logs:
            self._call_logs[-1].append(text)

    def discard_last_call(self) -> bool:
        if not self._call_logs:
            return False
        last_lines = self._call_logs.pop()
        for _ in reversed(last_lines):
            sys.stdout.write("\033[1A\033[2K")
        sys.stdout.flush()
        return True


def save_session(messages: Sequence[Any], path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    data = {
        "version": _SESSION_FILE_VERSION,
        "messages": messages_to_dict(list(messages)),
    }
    if metadata:
        data["metadata"] = metadata
    path_obj = Path(path)
    if path_obj.parent:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(path: str) -> Tuple[List[Any], Dict[str, Any]]:
    path_obj = Path(path)
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    version = payload.get("version") or 1
    if version not in (1, _SESSION_FILE_VERSION):
        raise ValueError(f"Unsupported session version {version!r}")
    message_data = payload.get("messages")
    if not isinstance(message_data, list):
        raise ValueError("Session file missing 'messages' list")
    metadata = payload.get("metadata") if version >= 2 else {}
    if metadata is None or not isinstance(metadata, dict):
        metadata = {}
    return list(messages_from_dict(message_data)), metadata


def _get_first_env(var_names: Sequence[str]) -> Optional[str]:
    for name in var_names:
        if not name:
            continue
        value = os.getenv(name)
        if value:
            return value
    return None


def _resolve_temperature(explicit: Optional[float]) -> float:
    if explicit is not None:
        return explicit
    for env_var in _GLOBAL_TEMP_ENV:
        env_value = os.getenv(env_var)
        if env_value is None:
            continue
        try:
            return float(env_value)
        except ValueError as exc:
            raise ValueError(f"{env_var} must be a float, got {env_value!r}") from exc
    return 0.0


def get_llm(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Construct a chat model for the requested provider.

    Args:
        provider: One of 'ollama', 'openai', 'deepseek', or 'openrouter'. Defaults to
            env var AGENT_PROVIDER (then AGENT_LLM_PROVIDER) or 'ollama'.
        model: Optional model override. Falls back to AGENT_MODEL, then AGENT_LLM_MODEL,
            then provider-specific env vars (e.g., AGENT_OPENAI_MODEL, OPENAI_MODEL), then a default.
        temperature: Optional override (otherwise AGENT_TEMPERATURE, AGENT_LLM_TEMPERATURE, or 0.0).
    """

    resolved_provider = (
        provider or _get_first_env(_GLOBAL_PROVIDER_ENV) or "ollama"
    ).lower()
    if resolved_provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(f"Unsupported LLM provider '{resolved_provider}'")

    resolved_temperature = _resolve_temperature(temperature)
    resolved_model = (
        model
        or _get_first_env(_GLOBAL_MODEL_ENV)
        or _get_first_env(_MODEL_ENV_VARS.get(resolved_provider, []))
        or _DEFAULT_MODELS[resolved_provider]
    )
    resolved_base_url = (
        _get_first_env(_GLOBAL_BASE_ENV)
        or _get_first_env(_BASE_ENV_VARS.get(resolved_provider, []))
        or _DEFAULT_BASE_URLS[resolved_provider]
    )

    if resolved_provider == "ollama":
        llm_kwargs = {
            "model": resolved_model,
            "temperature": resolved_temperature,
        }
        if resolved_base_url:
            llm_kwargs["base_url"] = resolved_base_url
        return ChatOllama(**llm_kwargs)

    provider_api_env = _PROVIDER_API_KEY_ENV_VARS.get(resolved_provider, [])
    if not provider_api_env:
        raise ValueError(f"No API key env vars defined for provider '{resolved_provider}'")
    resolved_api_key = (
        api_key
        or _get_first_env(_GLOBAL_API_KEY_ENV)
        or _get_first_env(provider_api_env)
    )
    if not resolved_api_key:
        tried_vars = [*_GLOBAL_API_KEY_ENV, *provider_api_env]
        tried_list = ", ".join(tried_vars)
        raise EnvironmentError(
            f"{resolved_provider} API key env var is not set; tried {tried_list}"
        )

    llm_kwargs = {
        "api_key": resolved_api_key,
        "model": resolved_model,
        "temperature": resolved_temperature,
    }
    if resolved_base_url:
        llm_kwargs["base_url"] = resolved_base_url

    return ChatOpenAI(**llm_kwargs)


def build_tool_map(tools: Sequence[Any]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for t in tools:
        name = getattr(t, "name", None)
        if not name:
            raise ValueError(f"Tool object {t} has no 'name' attribute")
        mapping[name] = t
    return mapping


def _normalize_tool_calls(ai_msg: AIMessage) -> Optional[List[Dict[str, Any]]]:
    tool_calls = getattr(ai_msg, "tool_calls", None)
    if tool_calls:
        return tool_calls

    # Legacy OpenAI-style function_call support (DeepSeek, older models)
    legacy_call = ai_msg.additional_kwargs.get("function_call") if hasattr(ai_msg, "additional_kwargs") else None
    if not legacy_call:
        return None

    legacy_args = legacy_call.get("arguments")
    if isinstance(legacy_args, str):
        try:
            legacy_args = json.loads(legacy_args)
        except Exception:
            legacy_args = {}
    elif legacy_args is None:
        legacy_args = {}

    return [
        {
            "id": f"legacy-{legacy_call.get('name', 'tool')}",
            "name": legacy_call.get("name"),
            "args": legacy_args,
        }
    ]


def _summarize_validation_error(exc: ValidationError) -> str:
    """
    Turn a verbose Pydantic ValidationError into a concise, agent-friendly summary.
    """
    try:
        errors = exc.errors()
    except Exception:
        return str(exc)

    messages: List[str] = []
    for err in errors:
        loc = err.get("loc") or ()
        path = ".".join(str(part) for part in loc if part is not None) or "value"
        msg = err.get("msg") or "Invalid value"
        received = err.get("input")
        snippet = repr(received)
        if isinstance(snippet, str) and len(snippet) > 80:
            snippet = snippet[:77] + "..."
        if received is not None:
            messages.append(f"{path}: {msg} (received {snippet})")
        else:
            messages.append(f"{path}: {msg}")
    return "; ".join(messages) or str(exc)


def run_agent_turn(
    llm_with_tools: Any,
    tools: Sequence[Any],
    messages: List[Any],
    max_tool_rounds: int = 32,
    tool_log: Optional[ToolCallLogBuffer] = None,
) -> str:
    """
    Run one conversational turn for the agent.

    The last message in 'messages' must be the user's HumanMessage.
    The model can perform tool->model cycles up to max_tool_rounds before returning
    a final natural language answer. 'messages' is mutated in place.
    """
    tool_map = build_tool_map(tools)

    def _log(line: str) -> None:
        if tool_log:
            tool_log.log_line(line)
        else:
            print(line)

    for _ in range(max_tool_rounds):
        ai_msg = llm_with_tools.invoke(messages)
        if not isinstance(ai_msg, AIMessage):
            content = str(getattr(ai_msg, "content", ai_msg))
            messages.append(AIMessage(content=content))
            return content

        messages.append(ai_msg)

        tool_calls = _normalize_tool_calls(ai_msg)
        invalid_calls = list(getattr(ai_msg, "invalid_tool_calls", []) or [])
        if not tool_calls:
            if invalid_calls:
                for invalid in invalid_calls:
                    error_payload = {
                        "ok": False,
                        "error": "invalid_tool_arguments",
                        "tool": invalid.get("name"),
                        "args": invalid.get("args"),
                        "message": invalid.get("error") or "Tool arguments could not be parsed.",
                    }
                    content_str = json.dumps(error_payload, indent=2)
                    _log(f"[tool-error] {invalid.get('name')} -> {content_str}")
                    messages.append(
                        ToolMessage(
                            content=content_str,
                            tool_call_id=invalid.get("id") or "",
                        )
                    )
                continue

            content = ai_msg.content or ""
            return str(content)

        # Execute each requested tool call and append ToolMessages
        for tc in tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {}) or {}
            tool_call_id = tc.get("id", "")
            if tool_log:
                tool_log.begin_call()

            if tool_name not in tool_map:
                error_payload = {
                    "ok": False,
                    "error": "unknown_tool",
                    "tool": tool_name,
                    "args": tool_args,
                    "message": f"Model requested unknown tool '{tool_name}'",
                }
                error_str = json.dumps(error_payload, indent=2)
                _log(f"[tool-call] {tool_name} args={tool_args}")
                _log(f"[tool-error] {tool_name} -> {error_str}")
                messages.append(
                    ToolMessage(
                        content=error_str,
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            tool_obj = tool_map[tool_name]

            _log(f"[tool-call] {tool_name} args={tool_args}")

            try:
                tool_result = tool_obj.invoke(tool_args)
                result_preview = str(tool_result)
                if len(result_preview) > 400:
                    result_preview = result_preview[:400] + "... [truncated]"
                _log(f"[tool-result] {tool_name} -> {result_preview}")
                content_str = str(tool_result)
            except ValidationError as exc:
                friendly_message = _summarize_validation_error(exc)
                error_payload = {
                    "ok": False,
                    "error": "invalid_tool_arguments",
                    "tool": tool_name,
                    "args": tool_args,
                    "message": friendly_message,
                }
                content_str = json.dumps(error_payload, indent=2)
                _log(f"[tool-error] {tool_name} -> {content_str}")
            except KeyboardInterrupt:
                cancel_payload = {
                    "ok": False,
                    "error": "cancelled_by_user",
                    "tool": tool_name,
                    "args": tool_args,
                    "message": "Tool execution halted by user interrupt.",
                }
                content_str = json.dumps(cancel_payload, indent=2)
                _log(f"[tool-error] {tool_name} -> {content_str}")
                messages.append(
                    ToolMessage(
                        content=content_str,
                        tool_call_id=tool_call_id,
                    )
                )
                raise
            except Exception as exc:
                error_payload = {
                    "ok": False,
                    "error": "tool_execution_error",
                    "tool": tool_name,
                    "args": tool_args,
                    "message": str(exc),
                }
                content_str = json.dumps(error_payload, indent=2)
                _log(f"[tool-error] {tool_name} -> {content_str}")

            messages.append(
                ToolMessage(
                    content=content_str,
                    tool_call_id=tool_call_id,
                )
            )

        # Next iteration will call the model again with ToolMessages included

    raise RuntimeError(
        f"Tool loop exceeded {max_tool_rounds} iterations without producing a final answer"
    )


def interactive_loop(
    llm_with_tools: Any,
    tools: Sequence[Any],
    system_prompt: str,
    *,
    intro_lines: Optional[List[str]] = None,
    initial_prompt: Optional[str] = None,
    exit_after_initial_prompt: bool = False,
    resume_path: Optional[str] = None,
    autosave_on_interrupt: bool = False,
    initial_messages: Optional[List[Any]] = None,
    resumed_session: bool = False,
    session_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    messages: List[Any] = list(initial_messages) if initial_messages else []
    metadata = session_metadata if session_metadata is not None else {}
    tool_log = ToolCallLogBuffer()
    agent_active = False

    if not messages:
        messages = [SystemMessage(content=system_prompt)]

    intro_lines = intro_lines or [
        "Agent session started.",
        "Type your questions or instructions.",
        "Ctrl+C while the agent is thinking cancels the current request; otherwise it saves and exits if configured.",
        "Commands: ':quit' to exit, '/delete' to remove the last tool call.",
    ]
    for line in intro_lines:
        print(line)

    if resume_path:
        if resumed_session:
            print(f"[session] Resumed conversation from {resume_path}. Ctrl+C will save progress.")
        else:
            print(f"[session] Session file: {resume_path} (will be created on exit).")

    def _maybe_save_session(reason: str) -> None:
        if not resume_path:
            return
        try:
            save_session(messages, resume_path, metadata)
            print(f"[session] Saved to {resume_path} ({reason}).")
        except Exception as exc:
            print(f"[session] Failed to save session to {resume_path}: {exc}")

    def _process_prompt(prompt_text: str) -> None:
        nonlocal agent_active
        start_len = len(messages)
        messages.append(HumanMessage(content=prompt_text))
        agent_active = True

        try:
            answer = run_agent_turn(llm_with_tools, tools, messages, tool_log=tool_log)
        except KeyboardInterrupt:
            agent_active = False
            if len(messages) == start_len + 1:
                del messages[start_len:]
            print("\n[session] Agent request cancelled.")
            return
        except Exception as exc:
            print(f"[error] {exc}")
            messages.append(AIMessage(content=f"Error occurred: {exc}"))
            agent_active = False
            return

        agent_active = False
        print(answer)
        messages.append(AIMessage(content=answer))

    def _delete_last_tool_call() -> None:
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if not isinstance(msg, ToolMessage):
                continue
            tool_call_id = getattr(msg, "tool_call_id", "") or ""
            del messages[idx]

            for j in range(idx - 1, -1, -1):
                prev_msg = messages[j]
                if not isinstance(prev_msg, AIMessage):
                    continue
                tc_list = list(prev_msg.tool_calls or [])
                if not tc_list:
                    continue
                removal_index: Optional[int] = None
                if tool_call_id:
                    for tc_idx in range(len(tc_list) - 1, -1, -1):
                        if tc_list[tc_idx].get("id") == tool_call_id:
                            removal_index = tc_idx
                            break
                else:
                    removal_index = len(tc_list) - 1
                if removal_index is None:
                    continue
                tc_list.pop(removal_index)
                prev_msg.tool_calls = tc_list
                if not tc_list and not prev_msg.content:
                    del messages[j]
                break

            cleared = tool_log.discard_last_call()
            if cleared:
                print("[session] Last tool call removed.")
            else:
                print("[session] Last tool call removed from context.")
            return

        print("[session] No tool calls to delete.")

    if initial_prompt:
        _process_prompt(initial_prompt)
        if exit_after_initial_prompt:
            if resume_path:
                _maybe_save_session("prompt complete")
            return

    try:
        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                print("\nExiting.")
                if resume_path:
                    _maybe_save_session("EOF")
                break
            except KeyboardInterrupt:
                if agent_active:
                    # Shouldn't happen; processing state handles interrupts
                    print("\n[session] Interrupt received; waiting for agent to finish.")
                    continue
                print("\nInterrupted. Saving and exiting.")
                if resume_path:
                    _maybe_save_session("keyboard interrupt")
                break

            if not user_input:
                continue
            if user_input == "/delete":
                _delete_last_tool_call()
                continue
            if user_input.lower() in (":q", ":quit", ":exit"):
                print("Goodbye.")
                if resume_path:
                    _maybe_save_session("quit command")
                break

            _process_prompt(user_input)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if autosave_on_interrupt:
            _maybe_save_session("keyboard interrupt")


__all__ = [
    "interactive_loop",
    "run_agent_turn",
    "build_tool_map",
    "get_llm",
    "save_session",
    "load_session",
]
