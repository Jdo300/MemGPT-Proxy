from types import SimpleNamespace

import pytest

from proxy_overlay import ProxyOverlayManager


class StubClient:
    def __init__(self):
        self.created = []
        self.modified = []
        self.attached = []

        class Blocks:
            def __init__(self, outer):
                self._outer = outer

            async def create(self, value: str, label: str, metadata=None, project_id=None):
                block_id = f"block_{len(self._outer.created) + 1}"
                self._outer.created.append((block_id, value, label, metadata, project_id))
                return SimpleNamespace(id=block_id)

            async def modify(self, block_id: str, value: str, label: str, project_id=None):
                self._outer.modified.append((block_id, value, label, project_id))
                return SimpleNamespace(id=block_id)

        class AgentBlocks:
            def __init__(self, outer):
                self._outer = outer

            async def attach(self, agent_id: str, block_id: str):
                self._outer.attached.append((agent_id, block_id))

        class Agents:
            def __init__(self, outer):
                self.blocks = AgentBlocks(outer)

        self.blocks = Blocks(self)
        self.agents = Agents(self)


class ErrorClient(StubClient):
    def __init__(self):
        super().__init__()
        self.failures = 0

        class BlocksWithError:
            def __init__(self, outer):
                self._outer = outer

            async def create(self, value: str, label: str, metadata=None, project_id=None):
                self._outer.failures += 1
                raise RuntimeError("boom")

            async def modify(self, block_id: str, value: str, label: str, project_id=None):
                self._outer.failures += 1
                raise RuntimeError("boom")

        self.blocks = BlocksWithError(self)


@pytest.mark.asyncio
async def test_overlay_creation_persists_block():
    client = StubClient()
    manager = ProxyOverlayManager(client, max_sessions=4, ttl_seconds=3600)

    overlay_changed, fallback = await manager.apply_overlay(
        "agent-1", "session-1", "system text", project_id="proj-1"
    )

    assert overlay_changed is True
    assert not fallback
    assert client.created[0][1] == "system text"
    assert client.created[0][-1] == "proj-1"
    assert client.attached[0] == ("agent-1", "block_1")

    state = manager.get_state("session-1")
    assert state.block_id == "block_1"
    assert state.overlay_hash


@pytest.mark.asyncio
async def test_overlay_update_is_idempotent():
    client = StubClient()
    manager = ProxyOverlayManager(client, max_sessions=4, ttl_seconds=3600)

    await manager.apply_overlay("agent-1", "session-1", "system text", project_id="proj-1")
    overlay_changed, fallback = await manager.apply_overlay(
        "agent-1", "session-1", "system text", project_id="proj-1"
    )

    assert overlay_changed is False
    assert not fallback
    assert not client.modified


@pytest.mark.asyncio
async def test_overlay_updates_existing_block():
    client = StubClient()
    manager = ProxyOverlayManager(client, max_sessions=4, ttl_seconds=3600)

    await manager.apply_overlay("agent-1", "session-1", "system text", project_id="proj-1")
    overlay_changed, _ = await manager.apply_overlay(
        "agent-1", "session-1", "new guidance", project_id="proj-1"
    )

    assert overlay_changed is True
    assert client.modified[0][0] == "block_1"
    assert client.modified[0][1] == "new guidance"
    assert client.modified[0][-1] == "proj-1"


@pytest.mark.asyncio
async def test_overlay_fallback_only_once():
    client = ErrorClient()
    manager = ProxyOverlayManager(client, max_sessions=4, ttl_seconds=3600)

    overlay_changed, fallback = await manager.apply_overlay(
        "agent-1", "session-1", "system text", project_id="proj-1"
    )
    assert overlay_changed is False
    assert len(fallback) == 1

    overlay_changed_again, fallback_again = await manager.apply_overlay(
        "agent-1", "session-1", "system text", project_id="proj-1"
    )
    assert overlay_changed_again is False
    assert fallback_again == []
    assert client.failures >= 2
