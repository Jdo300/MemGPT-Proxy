import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from letta_client import AsyncLetta, MessageCreate
from letta_client.types import TextContent

logger = logging.getLogger(__name__)


@dataclass
class SessionOverlayState:
    """Runtime state tracked for a single session's proxy overlay."""

    overlay_hash: Optional[str] = None
    block_id: Optional[str] = None
    fallback_applied: bool = False
    last_updated: float = field(default_factory=time.monotonic)


class _TTLCache:
    """Simple TTL + capacity bound cache supporting touch semantics."""

    def __init__(self, max_entries: int, ttl_seconds: int) -> None:
        self._store: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[str]:
        now = time.monotonic()
        value = self._store.get(key)
        if not value:
            return None
        timestamp, item = value
        if now - timestamp > self._ttl_seconds:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        self._store[key] = (now, item)
        return item

    def set(self, key: str, value: str) -> None:
        now = time.monotonic()
        self._store[key] = (now, value)
        self._store.move_to_end(key)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    def items(self) -> List[Tuple[str, str]]:
        now = time.monotonic()
        valid_items: List[Tuple[str, str]] = []
        to_delete: List[str] = []
        for key, (ts, value) in self._store.items():
            if now - ts <= self._ttl_seconds:
                valid_items.append((key, value))
            else:
                to_delete.append(key)
        for key in to_delete:
            del self._store[key]
        return valid_items


class SessionOverlayStore:
    """LRU/TTL store for overlay session state."""

    def __init__(self, max_sessions: int = 100, ttl_seconds: int = 3 * 60 * 60) -> None:
        self._store: "OrderedDict[str, SessionOverlayState]" = OrderedDict()
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds

    def get(self, session_id: str) -> Optional[SessionOverlayState]:
        now = time.monotonic()
        state = self._store.get(session_id)
        if not state:
            return None
        if now - state.last_updated > self._ttl_seconds:
            del self._store[session_id]
            return None
        self._store.move_to_end(session_id)
        return state

    def set(self, session_id: str, state: SessionOverlayState) -> None:
        state.last_updated = time.monotonic()
        self._store[session_id] = state
        self._store.move_to_end(session_id)
        while len(self._store) > self._max_sessions:
            self._store.popitem(last=False)

    def items(self) -> List[Tuple[str, SessionOverlayState]]:
        now = time.monotonic()
        valid_items: List[Tuple[str, SessionOverlayState]] = []
        to_delete: List[str] = []
        for session_id, state in self._store.items():
            if now - state.last_updated <= self._ttl_seconds:
                valid_items.append((session_id, state))
            else:
                to_delete.append(session_id)
        for session_id in to_delete:
            del self._store[session_id]
        return valid_items


class ProxyOverlayManager:
    """Manages the lifecycle of the "Proxy System Overlay" block for sessions."""

    OVERLAY_LABEL = "Proxy System Overlay"

    def __init__(
        self,
        client: AsyncLetta,
        *,
        max_sessions: int = 100,
        ttl_seconds: int = 3 * 60 * 60,
    ) -> None:
        self._client = client
        self._session_store = SessionOverlayStore(max_sessions=max_sessions, ttl_seconds=ttl_seconds)
        self._derived_sessions = _TTLCache(max_entries=max_sessions, ttl_seconds=ttl_seconds)

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def derive_session_id(self, agent_id: str, system_content: Optional[str], headers: Dict[str, str]) -> str:
        header_id = headers.get("x-session-id") or headers.get("X-Session-Id")
        if header_id:
            return header_id
        if system_content:
            content_hash = self._hash_content(system_content)
            key = f"{agent_id}:{content_hash}"
            cached = self._derived_sessions.get(key)
            if cached:
                return cached
            self._derived_sessions.set(key, key)
            return key
        fallback_id = f"{agent_id}:default"
        cached = self._derived_sessions.get(fallback_id)
        if cached:
            return cached
        self._derived_sessions.set(fallback_id, fallback_id)
        return fallback_id

    def get_state(self, session_id: str) -> Optional[SessionOverlayState]:
        return self._session_store.get(session_id)

    async def apply_overlay(
        self,
        agent_id: str,
        session_id: str,
        system_content: Optional[str],
        *,
        project_id: Optional[str] = None,
    ) -> Tuple[bool, List[MessageCreate]]:
        overlay_changed = False
        fallback_messages: List[MessageCreate] = []
        state = self._session_store.get(session_id) or SessionOverlayState()

        if not system_content:
            self._session_store.set(session_id, state)
            return overlay_changed, fallback_messages

        overlay_hash = self._hash_content(system_content)

        # Only return early if we have both the right hash AND a valid block
        # This prevents the bug where hash matches but block creation failed
        if state.overlay_hash == overlay_hash and state.block_id and state.block_id != "":
            self._session_store.set(session_id, state)
            return overlay_changed, fallback_messages

        try:
            if state.block_id:
                await self._client.blocks.modify(
                    state.block_id,
                    value=system_content,
                    label=self.OVERLAY_LABEL,
                    project_id=project_id,
                )
            else:
                block = await self._client.blocks.create(
                    value=system_content,
                    label=self.OVERLAY_LABEL,
                    metadata={"proxy_overlay_session": session_id},
                    project_id=project_id,
                )
                if not getattr(block, "id", None):
                    raise RuntimeError("Created overlay block missing id")
                await self._client.agents.blocks.attach(agent_id, block.id)
                state.block_id = block.id

            state.overlay_hash = overlay_hash
            state.fallback_applied = False
            overlay_changed = True
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Proxy overlay update failed for agent %s (session %s): %s", agent_id, session_id, exc
            )
            if not state.fallback_applied:
                fallback_messages.append(
                    MessageCreate(
                        role="user",
                        content=[TextContent(text=f"[Proxy System Overlay]: {system_content}")],
                    )
                )
                state.fallback_applied = True
            state.overlay_hash = overlay_hash
        finally:
            self._session_store.set(session_id, state)

        return overlay_changed, fallback_messages

    def debug_dump(self) -> Dict[str, Dict[str, Optional[str]]]:
        sessions: Dict[str, Dict[str, Optional[str]]] = {}
        for session_id, state in self._session_store.items():
            sessions[session_id] = {
                "overlay_hash": state.overlay_hash,
                "block_id": state.block_id,
                "fallback_applied": state.fallback_applied,
                "last_updated": state.last_updated,
            }
        derived = {key: value for key, value in self._derived_sessions.items()}
        return {"sessions": sessions, "derived_session_keys": derived}
