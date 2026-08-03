
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_k):
        return False

from FAIRGAME.src.llm_connectors.local_vllm_connector import LocalVLLMConnector

# Load environment variables from a .env file when python-dotenv is installed
load_dotenv()

# Lazy import helpers for API connectors (avoid ModuleNotFoundError on Kaggle)
def _get_anthropic_connector():
    from FAIRGAME.src.llm_connectors.anthropic_connector import AnthropicConnector
    return AnthropicConnector

def _get_mistral_connector():
    from FAIRGAME.src.llm_connectors.mistral_connector import MistralConnector
    return MistralConnector

def _get_openai_connector():
    from FAIRGAME.src.llm_connectors.openai_connector import OpenAIConnector
    return OpenAIConnector

# Dictionary mapping our abstract model names to provider classes and their corresponding provider model identifiers.
# API connectors use lazy loaders to avoid import errors when their SDKs are not installed.
MODEL_PROVIDER_MAP = {
    "Claude35Haiku": (_get_anthropic_connector, "claude-3-5-haiku-20241022"),
    "MistralLarge": (_get_mistral_connector, "mistral-large-latest"),
    "OpenAIGPT4o": (_get_openai_connector, "gpt-4o"),
    "OpenAIGPT5Nano": (_get_openai_connector, "gpt-5.4-nano"),
    # Local models for offline/Kaggle use (model path set via init_local_llm())
    "LocalGemma": (LocalVLLMConnector, "local-gemma"),
    "LocalLlama": (LocalVLLMConnector, "local-llama"),
    "LocalMistral": (LocalVLLMConnector, "local-mistral"),
    "LocalQwen": (LocalVLLMConnector, "local-qwen"),
    "LocalModel": (LocalVLLMConnector, "local-model"),
    # Add more mappings as needed.
}

class ChatModelFactory:
    """
    Factory for creating chat model instances based on the model name.
    """

    @staticmethod
    def get_model(model_name: str):
        """
        Return an instance of ChatModel based on the provided model name.

        Parameters:
            model_name (str): The abstract model name (e.g., "MistralLarge", "GPT4", "Claude").

        Returns:
            ChatModel: An instance of the appropriate chat model.

        Raises:
            ValueError: If the model_name is unsupported.
        """
        provider_info = MODEL_PROVIDER_MAP.get(model_name)
        if not provider_info:
            raise ValueError(f"Unsupported model specified: {model_name}")
        model_class_or_loader, provider_model = provider_info
        # If it's a lazy loader function (not a class), call it to get the actual class
        if callable(model_class_or_loader) and not isinstance(model_class_or_loader, type):
            model_class = model_class_or_loader()
        else:
            model_class = model_class_or_loader
        return model_class(provider_model)


def execute_prompt(model_name: str, prompt: str) -> str:
    """
    Execute a prompt using the specified model.

    Parameters:
        model_name (str): The abstract model name (e.g., "MistralLarge", "GPT4", "Claude").
        prompt (str): The prompt text to send to the API.

    Returns:
        str: The response text from the API.
    """
    chat_model = ChatModelFactory.get_model(model_name)
    return chat_model.send_prompt(prompt)


if __name__ == "__main__":
    # Example usage:
    model_identifier = "Claude35Sonnet"
    prompt_text = "Tell me a programming joke."
    
    try:
        result = execute_prompt(model_identifier, prompt_text)
        print("API Response:", result)
    except Exception as e:
        print("An error occurred:", e)
