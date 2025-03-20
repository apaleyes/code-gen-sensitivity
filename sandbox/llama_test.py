from models.deepseek import DeepSeek
from models.llama import Llama, LocalLlama, install_ollama

# Ensure API key is set before running the test
if "OPENROUTER_API_KEY" not in os.environ:
    raise ValueError(
        "OPENROUTER_API_KEY is not set. Please set it and restart your terminal."
    )

# Instantiate the Llama model
llama_model = LocalLlama()
ds_model = DeepSeek()

# Test prompt
test_prompt = "What is the capital of France?"

# Run the model
response = llama_model(test_prompt)
print("Response:", response)

# install_ollama()
