import logging
from typing import Dict

import os
import requests

BASE_URL = os.getenv("PROXY_URL", "http://localhost:8000/v1")

# Tool schema exposing a simple calculator function
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate basic math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2+2'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


def run_tool_test(prompt: str) -> None:
    logging.info("User: %s", prompt)
    payload: Dict[str, any] = {
        "model": "Milo",
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
    }
    # Allow generous time for proxy/Letta round-trips when tools are initialized
    response = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=180)
    logging.info("Proxy response: %s", response.text)
    if response.status_code != 200:
        logging.error("Request failed with status %s", response.status_code)
        return
    try:
        data = response.json()
    except Exception as exc:
        logging.error("JSON parse error: %s", exc)
        return
    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        call = tool_calls[0]
        logging.info("Tool call: %s(%s)", call["function"]["name"], call["function"]["arguments"])
    logging.info("Assistant: %s\n", message.get("content"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        "Use the calculator tool to compute 2+2.",
        "Say hello without using any tools.",
    ]
    for t in tests:
        logging.info("%s", "=" * 60)
        run_tool_test(t)


if __name__ == "__main__":
    main()
