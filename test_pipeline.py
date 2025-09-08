"""Mock test for letta_pipeline.py"""

from letta_pipeline import Pipeline


def run_test():
    pipeline = Pipeline()
    messages = [{"role": "user", "content": "Hello"}]
    # Use valves defaults for configuration
    stream = pipeline.pipe(
        user_message="Hello",
        model_id=pipeline.valves.agent_name,
        messages=messages,
        body={},
    )
    outputs = []
    for chunk in stream:
        outputs.append(chunk)
        print("STREAM:", chunk)
    assert any(isinstance(c, str) for c in outputs), "No assistant output received"


if __name__ == "__main__":
    run_test()
