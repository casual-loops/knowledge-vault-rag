from knowledge_rag.generation import (
    DeterministicGenerationProvider,
    GenerationProvider,
)
from knowledge_rag.generation_factory import get_generation_provider


def test_deterministic_generation_provider_returns_expected_response() -> None:
    provider = DeterministicGenerationProvider()

    result = provider.generate(
        prompt="Synthetic prompt",
    )

    assert result == "Generated response for: Synthetic prompt"


def test_generation_provider_factory_returns_provider() -> None:
    provider = get_generation_provider()

    assert isinstance(provider, GenerationProvider)


def test_generation_provider_factory_uses_deterministic_provider() -> None:
    provider = get_generation_provider()

    assert isinstance(
        provider,
        DeterministicGenerationProvider,
    )


from types import SimpleNamespace

from knowledge_rag.generation import (
    DeterministicGenerationProvider,
    GenerationProvider,
    OpenAIGenerationProvider,
)
from knowledge_rag.generation_factory import get_generation_provider


def test_openai_generation_provider_returns_output_text() -> None:
    calls: list[dict[str, str]] = []

    def fake_response_creator(
        *,
        model: str,
        input: str,
    ) -> SimpleNamespace:
        calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        return SimpleNamespace(
            output_text="Synthetic generated answer"
        )

    provider = OpenAIGenerationProvider(
        api_key="test-key",
        model="test-model",
        response_creator=fake_response_creator,
    )

    result = provider.generate(
        prompt="Synthetic prompt",
    )

    assert result == "Synthetic generated answer"

    assert calls == [
        {
            "model": "test-model",
            "input": "Synthetic prompt",
        }
    ]


def test_generation_provider_factory_uses_openai_when_configured(
    monkeypatch,
) -> None:
    import knowledge_rag.generation_factory as generation_factory

    monkeypatch.setattr(
        generation_factory.settings,
        "openai_api_key",
        "test-key",
    )

    monkeypatch.setattr(
        generation_factory.settings,
        "llm_model",
        "test-model",
    )

    provider = generation_factory.get_generation_provider()

    assert isinstance(
        provider,
        OpenAIGenerationProvider,
    )