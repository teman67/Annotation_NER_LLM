# llm_clients.py

import os
import openai
from typing import Optional
import openai
from openai import AuthenticationError, OpenAIError

try:
    import anthropic
except ImportError:
    anthropic = None

class LLMClient:
    def __init__(self, api_key: str, provider: str = "OpenAI", model: str = "gpt-4o"):
        self.api_key = api_key
        self.provider = provider
        self.model = model

        if self.provider == "OpenAI":
            openai.api_key = self.api_key
        elif self.provider == "Claude":
            if anthropic is None:
                raise ImportError("Anthropic SDK not installed. Run `pip install anthropic`")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1000) -> str:
        if self.provider == "OpenAI":
            return self._call_openai(prompt, temperature, max_tokens)
        elif self.provider == "Claude":
            return self._call_claude(prompt, temperature, max_tokens)

    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except AuthenticationError:
            raise ValueError("OpenAI API key invalid or unauthorized.")
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")
        except Exception as e:
            raise RuntimeError(f"OpenAI API unexpected error: {e}")

    def _call_claude(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Calls Claude API. Anthropic API uses tokens differently:
        max_tokens refers to total tokens in the completion.
        """
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=(
                    anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT
                ),
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            return response.completion.strip()
        except Exception as e:
            # Optionally check error string if needed
            if "authentication" in str(e).lower():
                raise ValueError("Anthropic API key invalid or unauthorized.")
            raise RuntimeError(f"Anthropic API error: {e}")
        # except Exception as e:
        #     raise RuntimeError(f"Anthropic API error: {e}")
