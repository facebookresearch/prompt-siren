# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from typing import overload

import httpx
from pydantic_ai import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client, Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        "Please install the `openai` package to use the Llama provider, "
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class PydanticAILlamaProvider(Provider[AsyncOpenAI]):
    """Provider for Llama API (https://llama.developer.meta.com/)."""

    @property
    def name(self) -> str:
        return "llama"

    @property
    def base_url(self) -> str:
        return "https://api.llama.com/compat/v1/"

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return OpenAIModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv("LLAMA_API_KEY")
        if not api_key and openai_client is None:
            raise UserError(
                "Set the `LLAMA_API_KEY` environment variable or pass it via `LlamaProvider(api_key=...)` "
                "to use the Llama API provider."
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client
            )
        else:
            http_client = cached_async_http_client(provider="llama")
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client
            )


class LlamaProvider:
    """Provider for Llama models.

    This provider handles all models with the "llama:" prefix and creates
    OpenAIChatModel instances configured with the Llama API client.

    Configuration is automatically read from the LLAMA_API_KEY environment variable.
    """

    def create_model(self, model_string: str) -> Model:
        """Create an OpenAIChatModel configured for Llama.

        Args:
            model_string: Full model string (e.g., "llama:Llama-3.3-8B-Instruct")

        Returns:
            OpenAIChatModel instance configured to work with the Llama API.
        """

        llama_provider = PydanticAILlamaProvider()
        model_name = model_string.split(":")[1]
        return OpenAIChatModel(model_name, provider=llama_provider)
