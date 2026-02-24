"""Unified Teacher client for frontier model APIs."""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from ..types.enums import TeacherProvider

logger = logging.getLogger(__name__)


class TeacherAPIError(Exception):
    """Error from Teacher API calls."""
    pass


class TeacherClient:
    """
    Unified client for Teacher model APIs.

    Supports:
    - OpenAI (GPT-4, GPT-4-Turbo)
    - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
    - Google (Gemini 2.0 Flash, Gemini 1.5 Pro)
    """

    def __init__(
        self,
        provider: TeacherProvider = TeacherProvider.GOOGLE,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback_providers: Optional[List[TeacherProvider]] = None,
    ):
        self.provider = provider
        self.model = model or self._default_model(provider)
        self.fallback_providers = fallback_providers or []
        self._clients: Dict[TeacherProvider, Any] = {}

        # Get API key from environment if not provided
        self._api_keys = {
            TeacherProvider.OPENAI: api_key or os.getenv('OPENAI_API_KEY'),
            TeacherProvider.ANTHROPIC: api_key or os.getenv('ANTHROPIC_API_KEY'),
            TeacherProvider.GOOGLE: api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'),
        }

        self._init_clients()

    def _default_model(self, provider: TeacherProvider) -> str:
        """Get default model for provider."""
        defaults = {
            TeacherProvider.OPENAI: 'gpt-4-turbo',
            TeacherProvider.ANTHROPIC: 'claude-3-5-sonnet-20241022',
            TeacherProvider.GOOGLE: 'gemini-2.0-flash',
        }
        return defaults.get(provider, 'gpt-4-turbo')

    def _init_clients(self):
        """Initialize API clients for available providers."""
        providers_to_init = [self.provider] + self.fallback_providers

        for provider in providers_to_init:
            api_key = self._api_keys.get(provider)
            if not api_key:
                logger.warning(f"No API key for {provider.value}, skipping initialization")
                continue

            try:
                if provider == TeacherProvider.OPENAI:
                    from openai import OpenAI
                    self._clients[provider] = OpenAI(api_key=api_key)

                elif provider == TeacherProvider.ANTHROPIC:
                    from anthropic import Anthropic
                    self._clients[provider] = Anthropic(api_key=api_key)

                elif provider == TeacherProvider.GOOGLE:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    self._clients[provider] = client

            except ImportError as e:
                logger.warning(f"Could not import client for {provider.value}: {e}")
            except Exception as e:
                logger.warning(f"Could not initialize {provider.value} client: {e}")

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from Teacher model.

        Args:
            prompt: User prompt
            system: System message (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Stop generation sequences

        Returns:
            Generated text

        Raises:
            TeacherAPIError: If all providers fail
        """
        providers = [self.provider] + self.fallback_providers
        last_error = None

        for provider in providers:
            if provider not in self._clients:
                continue

            try:
                return self._generate_with_retry(
                    provider=provider,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_sequences=stop_sequences
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.value} failed: {e}")
                continue

        raise TeacherAPIError(f"All providers failed. Last error: {last_error}")

    def _generate_with_retry(
        self,
        provider: TeacherProvider,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
        max_retries: int = 3
    ) -> str:
        """Generate with retry logic."""
        last_error = None

        for attempt in range(max_retries):
            try:
                if provider == TeacherProvider.OPENAI:
                    return self._openai_generate(
                        prompt, system, temperature, max_tokens, stop_sequences
                    )
                elif provider == TeacherProvider.ANTHROPIC:
                    return self._anthropic_generate(
                        prompt, system, temperature, max_tokens, stop_sequences
                    )
                elif provider == TeacherProvider.GOOGLE:
                    return self._google_generate(
                        prompt, system, temperature, max_tokens, stop_sequences
                    )
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retry {attempt + 1}/{max_retries} in {wait_time}s")
                    time.sleep(wait_time)

        raise last_error

    def _openai_generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Generate using OpenAI API."""
        client = self._clients[TeacherProvider.OPENAI]
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences
        )

        return response.choices[0].message.content

    def _anthropic_generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Generate using Anthropic API."""
        client = self._clients[TeacherProvider.ANTHROPIC]

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        if system:
            kwargs["system"] = system
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _google_generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Generate using Google GenAI SDK."""
        from google.genai import types

        client = self._clients[TeacherProvider.GOOGLE]

        # Build generation config
        config_kwargs = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
        }
        if system:
            config_kwargs['system_instruction'] = system
        if stop_sequences:
            config_kwargs['stop_sequences'] = stop_sequences

        config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        return response.text
