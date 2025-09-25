#!/usr/bin/env python
"""Simple smoke test covering proxy overlay behavior."""

import json
import os
import sys
import uuid
from typing import Iterable

import requests

BASE_URL = os.getenv("PROXY_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("PROXY_MODEL", "Milo")
SESSION_ID = os.getenv("PROXY_SESSION_ID", uuid.uuid4().hex)


def _request(messages, *, stream=False):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": stream,
    }
    headers = {"X-Session-Id": SESSION_ID}
    url = f"{BASE_URL}/v1/chat/completions"
    if stream:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        return response.iter_lines()
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def _print(title: str, detail: str) -> None:
    print(f"[{title}] {detail}")


def main() -> int:
    system_message = {
        "role": "system",
        "content": "You are Milo, a concise assistant for overlay validation.",
    }
    first_user = {"role": "user", "content": "Hello overlay!"}
    follow_up_user = {"role": "user", "content": "Second turn, no system."}
    changed_system = {
        "role": "system",
        "content": "You are Milo, now respond enthusiastically!",
    }
    final_user = {"role": "user", "content": "Give me a pep talk."}

    try:
        resp1 = _request([system_message, first_user])
        _print("step1", f"status=ok choices={len(resp1['choices'])}")

        resp2 = _request([follow_up_user])
        _print("step2", f"status=ok overlay_unchanged expected choices={len(resp2['choices'])}")

        resp3 = _request([changed_system, follow_up_user])
        _print("step3", f"status=ok overlay_updated choices={len(resp3['choices'])}")

        stream_lines = _request([final_user], stream=True)
        chunk_count = 0
        for raw_line in stream_lines:
            if not raw_line:
                continue
            if raw_line == b"data: [DONE]":
                break
            if raw_line.startswith(b"data: "):
                chunk_count += 1
        _print("step4", f"stream_chunks={chunk_count}")
    except requests.HTTPError as exc:
        _print("error", f"HTTP failure: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - just diagnostic
        _print("error", f"Unexpected failure: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
