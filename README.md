# MemGPT-Proxy
Python script that creates a Proxy server to translate the MemGPT server's client protocol into a standard OpenAI ChatCompletion format with additional parameters provided to get the agent's inner thoughts and function calls executed.

# Setup Instructions
Simply change the `API_TOKEN` and `MEMGPT_BASE_URL` values to your MemGPT API Server's Token and URL and run the script. Then you should be able to access your MemGPT agents as if they were OpenAI models. Use the name of your MemGPT agent in the model field when making requests. 
