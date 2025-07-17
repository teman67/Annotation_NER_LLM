# llm_clients.py
import streamlit as st
import os
import openai
from typing import Optional
from openai import AuthenticationError, OpenAIError
try:
    import anthropic
except ImportError:
    anthropic = None

class LLMClient:
    def __init__(self, api_key, provider, model):  # Fixed: __init__ instead of **init**
        self.api_key = api_key
        self.provider = provider
        self.model = model
       
    def generate(self, prompt, temperature=0.1, max_tokens=1000):
        """
        Enhanced generate method with better error handling
        """
        try:
            if not prompt or prompt.strip() == "":
                raise ValueError("Empty prompt provided")
               
            if self.provider == "OpenAI":
                return self._call_openai(prompt, temperature, max_tokens)
            elif self.provider == "Claude":
                return self._call_claude(prompt, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
               
        except Exception as e:
            st.error(f"❌ LLM API call failed: {e}")
            return None  # <--- Return None instead of "[]"
   
    def _call_openai(self, prompt, temperature, max_tokens):  # Fixed: _call_openai instead of *call*openai
        import openai
       
        # The rest of the codes are removed to make it private.
               
            return content.strip()
           
        except openai.RateLimitError:
            st.error("OpenAI rate limit exceeded. Please wait and try again.")
            return None
        except openai.APITimeoutError:
            st.error("OpenAI API timeout. Please try again.")
            return None
        except openai.AuthenticationError:
            st.error("OpenAI authentication failed. Please check your API key. Have a look at the [OpenAI API Key Guide](https://platform.openai.com/api-keys) for more details.")
            return None
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
            return None
   
    def _call_claude(self, prompt, temperature, max_tokens):  # Fixed: _call_claude instead of *call*claude
        if anthropic is None:
            st.error("Anthropic library not installed. Please install it with: pip install anthropic")
            return "[]"
       
        # The rest of the codes are removed to make it private.
            return content.strip()
           
        except anthropic.RateLimitError:
            st.error("Claude rate limit exceeded. Please wait and try again.")
            return None
        except anthropic.APITimeoutError:
            st.error("Claude API timeout. Please try again.")
            return None
        except anthropic.AuthenticationError:
            st.error("Claude authentication failed. Please check your API key. Have a look at the [Claude API Key Guide](https://docs.anthropic.com/en/api/overview) for more details.")
            return None
        except Exception as e:
            st.error(f"Claude API error: {e}")
            return None