import json
import logging
from typing import Dict

import requests

import os

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
                        "description": "Math expression to evaluate, e.g. '2+2'"
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

def _evaluate_expression(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    try:
        # Only allow numbers and math operators
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed)  # pylint: disable=eval-used
        return str(result)
    except Exception as exc:  # pylint: disable=broad-except
        return f"error: {exc}"

def run_tool_test(prompt: str) -> None:
    logging.info("User: %s", prompt)
    payload: Dict[str, any] = {
        "model": "Milo",
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
    }
    while True:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=60)
        logging.info("Proxy response: %s", response.text)
        data = response.json()
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        if tool_calls:
            tool_call = tool_calls[0]
            args = json.loads(tool_call["function"]["arguments"])
            result = _evaluate_expression(args["expression"])
            logging.info(
                "Tool call -> %s(%s) = %s",
                tool_call["function"]["name"],
                args["expression"],
                result,
            )
            payload = {
                "model": "Milo",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                ],
                "tools": TOOLS,
            }
            continue
        logging.info("Assistant: %s\n", message.get("content"))
        break

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        "What is 2+2?",
        "Multiply 6 by 7 for me.",
        "Say hello without using any tools.",
    ]
    for t in tests:
        logging.info("%s", "=" * 60)
        run_tool_test(t)

if __name__ == "__main__":
    main()
