from openai import OpenAI, RateLimitError
import os
import time

from FAIRGAME.src.llm_connectors.abstract_connector import AbstractConnector

class OpenAIConnector(AbstractConnector):
    """
    Chat model implementation for the OpenAI API.
    """

    def __init__(self, provider_model: str, temperature: float = 1.0):
        self.api_key = os.getenv("API_KEY_OPENAI")
        if not self.api_key:
            raise EnvironmentError("API_KEY_OPENAI not found in environment variables.")
        self.provider_model = provider_model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def send_prompt(self, prompt: str, max_retries: int = 8) -> str:
        messages = [{"role": "user", "content": prompt}]
        # Tài khoản free/tier thấp có TPM (tokens-per-minute) rất hẹp -> chạy batch
        # nhiều game sẽ va 429 rate_limit_exceeded giữa chừng. Retry với backoff tăng
        # dần thay vì để crash cả run (không mất progress đã chạy).
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.provider_model,
                    temperature=self.temperature,
                    messages=messages
                )
                return completion.choices[0].message.content
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 60))
