import os
import time
import google.genai as genai
from google.genai import types
from typing import Dict, Any, Optional
from engine.core.config import GOOGLE_API_KEY
from engine.core.llm_provider import LLMProvider

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.client = genai.Client(api_key=self.api_key)
        self.model = model_name

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 1) -> Dict[str, Any]:
        start_time = time.time()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config= types.GenerateContentConfig(
                system_instruction= system_prompt if system_prompt else "",
                temperature=temperature
            )
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # Gemini usage data is in response.usage_metadata
        content = response.text

        return {
            "content": content,
            "latency_ms": latency_ms,
            "provider": "google"
        }
    
if __name__ == "__main__":
    # Test GeminiProvider with a sample prompt
    provider = GeminiProvider(model_name="gemini-2.5-flash-lite", api_key=GOOGLE_API_KEY)
    test_prompt = "What is gravity?"

    result = provider.generate(test_prompt)
    print("Generated Content:", result["content"])
    
