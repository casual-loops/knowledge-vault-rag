from abc import ABC, abstractmethod
from typing import Any, Protocol

from openai import OpenAI


class ResponseCreator(Protocol):
    """Callable shape required for generation."""

    def __call__(
        self,
        *,
        model: str,
        input: str,
    ) -> Any:
        ...


class GenerationProvider(ABC):
    """Interface implemented by answer-generation providers."""

    @abstractmethod
    def generate(
        self,
        *,
        prompt: str,
    ) -> str:
        """Generate a response for the supplied prompt."""

    @property
    @abstractmethod
    def is_external(self) -> bool:
        """Whether this provider sends content outside the local environment."""


class DeterministicGenerationProvider(GenerationProvider):
    """Deterministic provider intended for tests and offline development."""

    def generate(
        self,
        *,
        prompt: str,
    ) -> str:
        return f"Generated response for: {prompt}"

    @property
    def is_external(self) -> bool:
        return False


class OpenAIGenerationProvider(GenerationProvider):
    """Generation provider backed by the OpenAI API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        response_creator: ResponseCreator | None = None,
    ) -> None:
        if response_creator is None:
            client = OpenAI(api_key=api_key)

            def create_response(
                *,
                model: str,
                input: str,
            ) -> Any:
                return client.responses.create(
                    model=model,
                    input=input,
                )

            self._response_creator: ResponseCreator = create_response
        else:
            self._response_creator = response_creator

        self._model = model

    def generate(
        self,
        *,
        prompt: str,
    ) -> str:
        response = self._response_creator(
            model=self._model,
            input=prompt,
        )

        return str(response.output_text)

    @property
    def is_external(self) -> bool:
        return True