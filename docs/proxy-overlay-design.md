# Proxy System Overlay Design

This document summarizes how the Letta Proxy handles system prompts and chat history when interacting with stateful Letta agents.

## Overlay memory block

* On the first request that includes a `system` message, the proxy stores the message content in a durable block named **"Proxy System Overlay"** using Letta's block APIs. The block is attached to the target agent without modifying the agent's core memory.
* Subsequent requests reuse the same block and are idempotent—if the system text has not changed, no API call is made.
* When the system text changes, the block is updated in place. Block metadata records the active proxy session identifier.
* If writing the overlay fails, the proxy logs a warning and injects the system text once as a chat preface for that session (fallback mode). The fallback is not repeated on later turns.

## Session tracking

* Clients can supply an explicit `X-Session-Id` header; otherwise the proxy derives a session identifier from `agent_id + sha256(system_message)` and caches it in an in-process LRU (capacity 100, TTL three hours).
* A per-session record tracks the overlay block id, last hash, and whether fallback mode has been used. Entries expire automatically after the TTL.
* An optional debug endpoint (`/debug/sessions`) can be enabled with the `PROXY_DEBUG_SESSIONS=1` environment variable. It returns the active session table and derived cache contents to aid troubleshooting.

## Delta-only chat forwarding

* For each `/v1/chat/completions` request the proxy forwards only the new conversation delta:
  * Any tool-return payloads supplied via `tool_results`.
  * Trailing `tool` messages after the most recent assistant turn (if the client embeds them in `messages`).
  * The final `user` message (if present).
* System messages are consumed solely for overlay management—they are **not** sent as regular chat turns once the overlay is in place.
* If the proxy cannot determine any outbound delta (e.g., overlay-only ping), it returns an empty assistant response immediately without contacting the Letta server.

## Streaming behaviour

* Streaming requests keep a stable stream id, send an initial assistant-role primer chunk, relay Letta's chunks verbatim (`ensure_ascii=False`), and finish with `data: [DONE]`.
* When no outbound delta exists for a streaming request, the proxy emits just the primer chunk followed by `[DONE]`.

## Smoke test script

Run `scripts/proxy_overlay_smoketest.py` to exercise the overlay pathway. The script issues four calls—initial system+user, a user-only follow-up, a system-change turn, and a streamed message—printing concise status lines for each step. Environment variables `PROXY_BASE_URL`, `PROXY_MODEL`, and `PROXY_SESSION_ID` can override the defaults.
