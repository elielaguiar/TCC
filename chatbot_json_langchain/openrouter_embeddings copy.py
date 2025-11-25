import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import APIError, OpenAI, RateLimitError

load_dotenv()


@dataclass
class OpenRouterConfig:
    api_key: str
    base_url: str
    http_referer: str
    x_title: str
    model: str
    batch_size: int = 32
    max_retries: int = 5
    retry_base: float = 2.0


class OpenRouterEmbeddings(Embeddings):
    """Embedding wrapper with explicit batching and retry handling for OpenRouter."""

    def __init__(self, config: OpenRouterConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": config.http_referer,
                "X-Title": config.x_title,
            },
        )

    @classmethod
    def from_env(cls) -> "OpenRouterEmbeddings":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        http_referer = os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_SITE_URL")
        x_title = os.getenv("OPENROUTER_X_TITLE") or os.getenv("OPENROUTER_APP_TITLE")
        model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
        batch_size = int(os.getenv("OPENROUTER_BATCH_SIZE", "32"))
        max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "5"))

        missing = []
        if not api_key:
            missing.append("OPENROUTER_API_KEY")
        if not http_referer:
            missing.append("OPENROUTER_HTTP_REFERER ou OPENROUTER_SITE_URL")
        if not x_title:
            missing.append("OPENROUTER_X_TITLE ou OPENROUTER_APP_TITLE")
        if missing:
            raise RuntimeError(
                "Variáveis de ambiente ausentes: " + ", ".join(missing)
            )

        config = OpenRouterConfig(
            api_key=api_key,
            base_url=base_url,
            http_referer=http_referer,
            x_title=x_title,
            model=model,
            batch_size=max(batch_size, 1),
            max_retries=max(max_retries, 1),
        )
        return cls(config)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return list(self._embed_batches(texts))

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def _embed_batches(self, texts: Sequence[str]) -> Iterable[List[float]]:
        for batch in _chunk_sequence(texts, self.config.batch_size):
            embeddings = self._request_embeddings(batch)
            for emb in embeddings:
                yield emb

    def _request_embeddings(self, batch: Sequence[str]) -> List[List[float]]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=list(batch),
                )
                if not response or not getattr(response, "data", None):
                    raise RuntimeError("Resposta vazia do provedor de embeddings")
                return [item.embedding for item in response.data]
            except RateLimitError as exc:
                last_error = exc
                sleep_time = self.config.retry_base ** attempt
                time.sleep(sleep_time)
            except APIError as exc:
                last_error = exc
                if getattr(exc, "status_code", 0) in {500, 502, 503, 504}:
                    sleep_time = self.config.retry_base ** attempt
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Erro da API do OpenRouter: {exc}") from exc
            except Exception as exc:  # pragma: no cover - superfície externa
                raise RuntimeError(f"Falha ao solicitar embeddings: {exc}") from exc
        raise RuntimeError(f"Falha após múltiplas tentativas: {last_error}")


def _chunk_sequence(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    if size <= 0:
        raise ValueError("size deve ser positivo")
    for start in range(0, len(seq), size):
        yield seq[start : start + size]
