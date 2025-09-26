"""
Proxy Overlay System for Letta Agents

This module implements the Proxy System Overlay functionality that enables persistent
system prompt management for Letta agents. Instead of injecting system prompts into
chat messages, the system stores them as persistent memory blocks in Letta's backend.

Key Features:
- **Dynamic Block Sizing**: Supports unlimited system prompt lengths (50K+ characters)
- **Read-Only Protection**: Prevents agents from modifying system prompts
- **Smart Block Reuse**: Checks for existing blocks to avoid database constraint violations
- **Session-Based Caching**: Efficient caching with TTL for session state management
- **Graceful Fallback**: Falls back to message-based system prompts if block creation fails
- **Content Hashing**: Uses SHA256 hashing to detect system prompt changes
- **LRU/TTL Caching**: Efficient memory management for session state

Architecture:
    ProxyOverlayManager: Main coordinator for overlay lifecycle management
    SessionOverlayStore: LRU/TTL cache for session state persistence
    _TTLCache: Generic TTL cache implementation for derived session keys

The system ensures that:
1. System prompts are stored as persistent Letta memory blocks
2. Changes to system prompts are detected via content hashing
3. Existing blocks are reused when content hasn't changed
4. New blocks are created with appropriate size limits for large prompts
5. Blocks are marked as read-only to prevent agent modification
6. Session state is cached efficiently with TTL expiration

Usage:
    overlay_manager = ProxyOverlayManager(letta_client)
    overlay_changed, fallbacks = await overlay_manager.apply_overlay(
        agent_id, session_id, system_content, project_id=project_id
    )

Author: Jason Owens
Version: 1.0.0
"""

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
    """Runtime state tracked for a single session's proxy overlay.

    This dataclass maintains the current state of system prompt overlay management
    for a specific session, including content hash, block ID, and fallback status.

    Attributes:
        overlay_hash: SHA256 hash of the current system content for change detection
        block_id: ID of the persistent memory block in Letta backend
        fallback_applied: Whether fallback message injection has been used for this session
        last_updated: Timestamp of last state update (monotonic time)
    """

    overlay_hash: Optional[str] = None
    block_id: Optional[str] = None
    fallback_applied: bool = False
    last_updated: float = field(default_factory=time.monotonic)


class _TTLCache:
    """Simple TTL + capacity bound cache supporting touch semantics.

    Generic cache implementation with time-to-live (TTL) expiration and maximum
    capacity limits. Uses LRU eviction when capacity is exceeded and automatically
    removes expired entries on access.

    Attributes:
        _store: OrderedDict storing (timestamp, value) pairs for LRU ordering
        _max_entries: Maximum number of entries to store before eviction
        _ttl_seconds: Time-to-live in seconds for cache entries

    Methods:
        get: Retrieve value if not expired, update access time
        set: Store value with current timestamp, evict if needed
        items: Return all valid (non-expired) key-value pairs
    """

    def __init__(self, max_entries: int, ttl_seconds: int) -> None:
        """Initialize TTL cache with capacity and expiration settings.

        Args:
            max_entries: Maximum number of entries before LRU eviction
            ttl_seconds: TTL for entries in seconds
        """
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
    """Manages the lifecycle of the "Proxy System Overlay" block for sessions.

    This is the main coordinator for system prompt overlay management in the Letta Proxy.
    It handles the creation, updating, and management of persistent memory blocks that
    store system prompts separately from chat messages.

    Key Features:
    - **Dynamic Block Management**: Creates and updates Letta memory blocks for system prompts
    - **Change Detection**: Uses SHA256 hashing to detect system prompt changes
    - **Smart Reuse**: Reuses existing blocks when content hasn't changed
    - **Session Caching**: Efficient caching of session state with TTL
    - **Fallback Support**: Graceful fallback to message-based system prompts if needed
    - **Read-Only Blocks**: Ensures system prompts cannot be modified by agents

    The manager maintains separate caches for:
    - Session overlay state (SessionOverlayStore)
    - Derived session keys (_TTLCache)

    Attributes:
        OVERLAY_LABEL: Standard label used for overlay blocks ("proxy_system_overlay")
        _client: AsyncLetta client for backend communication
        _session_store: Session state management with LRU/TTL
        _derived_sessions: Cache for derived session keys

    Args:
        client: AsyncLetta client instance
        max_sessions: Maximum sessions to cache (default: 100)
        ttl_seconds: TTL for cached sessions in seconds (default: 3 hours)
    """

    OVERLAY_LABEL = "proxy_system_overlay"

    def __init__(
        self,
        client: AsyncLetta,
        *,
        max_sessions: int = 100,
        ttl_seconds: int = 3 * 60 * 60,
    ) -> None:
        """Initialize the proxy overlay manager.

        Args:
            client: AsyncLetta client for backend communication
            max_sessions: Maximum number of sessions to cache
            ttl_seconds: Time-to-live for cached sessions (3 hours default)
        """
        self._client = client
        self._session_store = SessionOverlayStore(max_sessions=max_sessions, ttl_seconds=ttl_seconds)
        self._derived_sessions = _TTLCache(max_entries=max_sessions, ttl_seconds=ttl_seconds)

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate SHA256 hash of content for change detection.

        Args:
            content: System content to hash

        Returns:
            SHA256 hex digest of the content
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def derive_session_id(self, agent_id: str, system_content: Optional[str], headers: Dict[str, str]) -> str:
        """Derive a unique session identifier for overlay management.

        Session IDs are derived from multiple sources in order of preference:
        1. Explicit session ID from headers (x-session-id or X-Session-Id)
        2. Content-based ID using system content hash (for content-specific sessions)
        3. Default fallback ID for agent (when no content is provided)

        This allows for flexible session management while ensuring consistent
        overlay behavior for the same content.

        Args:
            agent_id: The Letta agent identifier
            system_content: System prompt content for hash-based session IDs
            headers: HTTP headers that may contain explicit session ID

        Returns:
            Unique session identifier string
        """
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

        clean_content = (
            system_content.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        )

        overlay_hash = self._hash_content(clean_content)

        # Only return early if we have both the right hash AND a valid block
        # This prevents the bug where hash matches but block creation failed
        if state.overlay_hash == overlay_hash and state.block_id and state.block_id != "":
            self._session_store.set(session_id, state)
            return overlay_changed, fallback_messages

        try:
            if state.block_id:
                logger.info(f"Updating existing block {state.block_id} for agent {agent_id}")
                await self._client.blocks.modify(
                    state.block_id,
                    value=clean_content,
                    label=self.OVERLAY_LABEL,
                    project_id=project_id,
                )
                logger.info(f"Successfully updated block {state.block_id}")
            else:
                # Check if a block with this label already exists for this agent
                existing_blocks: List = []
                list_method = getattr(self._client.agents.blocks, "list", None)
                if callable(list_method):
                    existing_blocks = await list_method(agent_id)
                overlay_block = None

                # Look for existing block with our overlay label
                for block in existing_blocks:
                    if hasattr(block, 'label') and block.label == self.OVERLAY_LABEL:
                        overlay_block = block
                        break

                if overlay_block:
                    # Update the existing block instead of creating a new one
                    logger.info(f"Updating existing overlay block {overlay_block.id} for agent {agent_id}")
                    await self._client.blocks.modify(
                        overlay_block.id,
                        value=clean_content,
                        label=self.OVERLAY_LABEL,
                        project_id=project_id,
                    )
                    state.block_id = overlay_block.id
                    logger.info(f"Successfully updated existing overlay block {overlay_block.id}")
                else:
                    logger.info(f"Creating new block for agent {agent_id} (session {session_id})")
                    logger.info(f"System content length: {len(clean_content)} characters")
                    logger.info(f"System content preview: {clean_content[:200]}...")
                    # Use the limit parameter to allow blocks up to the actual content size
                    # This should override the default 20K limit
                    content_limit = len(clean_content)

                    logger.info(f"Creating block with limit={content_limit} for {len(clean_content)} character content")

                    base_kwargs = {
                        "value": clean_content,
                        "label": self.OVERLAY_LABEL,
                        "metadata": {"proxy_overlay_session": session_id},
                        "project_id": project_id,
                    }

                    try:
                        # Create the block with the dynamic limit and read-only flag
                        block = await self._client.blocks.create(
                            **base_kwargs,
                            limit=content_limit,  # Set limit to match actual content size
                            read_only=True,  # Lock the block so agents can't modify it
                        )
                    except TypeError as type_error:
                        error_message = str(type_error)
                        if "unexpected keyword argument" in error_message:
                            block = await self._client.blocks.create(**base_kwargs)
                        else:
                            raise

                    # Extract block ID (SDK returns letta_client.types.block.Block object)
                    block_id = block.id
                    if not block_id:
                        raise RuntimeError("Created overlay block missing id")

                    # Attach block to agent
                    await self._client.agents.blocks.attach(agent_id, block_id)

                    state.block_id = block_id

            state.overlay_hash = overlay_hash
            state.fallback_applied = False
            overlay_changed = True
            logger.info(f"Overlay successfully applied for agent {agent_id} (session {session_id})")
        except Exception as exc:  # pragma: no cover
            logger.error(
                f"Proxy overlay update failed for agent {agent_id} (session {session_id}): {exc}",
                exc_info=True  # Include full traceback
            )
            logger.error(f"Exception type: {type(exc)}")
            if not state.fallback_applied:
                logger.warning(f"Applying fallback message for agent {agent_id}")
                fallback_messages.append(
                    MessageCreate(
                        role="user",
                        content=[TextContent(text=f"[Proxy System Overlay]: {clean_content}")],
                    )
                )
                state.fallback_applied = True
            state.overlay_hash = overlay_hash
        finally:
            self._session_store.set(session_id, state)

        return overlay_changed, fallback_messages

    def debug_dump(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Debug method to inspect internal state of overlay manager.

        Provides comprehensive debugging information about active sessions,
        including overlay state, block IDs, and cached derived session keys.
        This is useful for troubleshooting overlay issues and understanding
        system behavior.

        Returns:
            Dictionary containing:
            - sessions: Mapping of session IDs to their current state
            - derived_session_keys: Cached derived session key mappings

        Note:
            This method is only available when PROXY_DEBUG_SESSIONS=1
        """
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
