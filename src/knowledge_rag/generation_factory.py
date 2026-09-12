from knowledge_rag.config import settings
from knowledge_rag.generation import (
    DeterministicGenerationProvider,
    GenerationProvider,
    OpenAIGenerationProvider,
)


def get_generation_provider() -> GenerationProvider:
    """Create the configured generation provider."""

    if settings.openai_api_key and settings.llm_model:
        return OpenAIGenerationProvider(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
        )

    return DeterministicGenerationProvider()