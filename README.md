# agent_runtime

Agent Runtime bundles the generic orchestration pieces for chat-style LLM agents:

- `get_llm` resolves providers/models/base URLs/API keys from flags or env vars (Ollama, OpenAI, DeepSeek, OpenRouter).
- `run_agent_turn` and `interactive_loop` implement the model ↔ tool cycle, including validation-friendly error payloads, `/delete` last tool-call support, and session autosave hooks.
- `save_session` / `load_session` serialize LangChain message streams (plus arbitrary metadata) so sessions can be resumed anywhere.

The module has zero reverse-engineering knowledge—callers supply a system prompt, tool bundle, and any intro text. Aire’s CLIs import it to keep their entrypoints thin, but any other agent can do the same (CLI, LangServe, orchestrators, etc.).

## Installation

From the repo root you can install it directly:

```bash
pip install ./agent_runtime
```

or in editable mode while developing:

```bash
pip install -e ./agent_runtime[test]
```

Usage in this repo:
- `aire_light/agent_cli.py` registers local binaries + r2 tools, then delegates to `interactive_loop`.
- `aire/agent_cli.py` provisions a workspace, uploads binaries, wires RPC tools, then runs the same loop.

Bring your own tools/system prompts to reuse the runtime elsewhere.
