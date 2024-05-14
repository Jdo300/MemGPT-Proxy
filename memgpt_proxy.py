from flask import Flask, request, jsonify
import logging
import uuid
import time
from memgpt import create_client
from types import SimpleNamespace

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# MemGPT Configuration
API_TOKEN = "{API KEY}"  # Replace with your actual MemGPT API token
MEMGPT_BASE_URL = 'http://localhost:8283'  # Replace with actual URL

# Create MemGPT client
memgpt_client = create_client(base_url=MEMGPT_BASE_URL, token=API_TOKEN)

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json(force=True)

        if 'model' not in data or 'messages' not in data:
            return jsonify({"error": "Missing required fields in the request"}), 400

        agent_name = data['model']
        input_messages = data['messages']

        logging.info(f"Request received for agent: {agent_name} with messages: {input_messages}")

        agent_id = get_memgpt_agent_id(agent_name)
        if not agent_id:
            return jsonify({"error": "Agent not found"}), 404

        # Build the prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in input_messages])

        # Send prompt to MemGPT and receive response
        memgpt_response = memgpt_client.user_message(agent_id=agent_id, message=prompt)

        # Process the response to structure it correctly
        formatted_choices = []
        for message in memgpt_response.messages:
            # Assuming each message in response contains 'content', 'internal_monologue', and 'function_call'
            choice = {
                "message": {
                    "role": "assistant",
                    "content": message.get('assistant_message', ''),
                    "memgpt_data": {
                        "internal_monologue": message.get('internal_monologue', ''),
                        "function_call": message.get('function_call', {})
                    }
                },
                "finish_reason": "stop"
            }
            formatted_choices.append(choice)

        # Create the final structured response
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": agent_name,
            "choices": formatted_choices,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": sum(len(choice['message']['content'].split()) for choice in formatted_choices),
                "total_tokens": len(prompt.split()) + sum(len(choice['message']['content'].split()) for choice in formatted_choices)
            }
        }

        logging.info(f"Response prepared: {response}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during request processing: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def get_memgpt_agent_id(agent_name: str) -> str:
    """
    Helper function to retrieve the MemGPT agent ID based on the agent name.
    Returns None if the agent is not found.
    """
    agents = memgpt_client.list_agents().agents
    for agent in agents:
        if agent['name'] == agent_name:
            return agent['id']
    return None

if __name__ == '__main__':
    app.run(debug=True)
