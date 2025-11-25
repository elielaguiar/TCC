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
    embedding_model: str
    batch_size: int = 32
    max_retries: int = 5
    retry_base: float = 2.0


class OpenRouterEmbeddings(Embeddings):
    """
    Embedding wrapper para usar OpenRouter via SDK openai>=1.0,
    com headers obrigatórios (HTTP-Referer e X-Title).
    """

    def __init__(self, config: OpenRouterConfig) -> None:
        self.config = config

        # Debug para confirmar carregamento das variáveis
        print("DEBUG API KEY:", config.api_key[:10] + "...")
        print("DEBUG BASE URL:", config.base_url)

        # IMPORTANTE: openai==2.2.0 exige default_headers
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers={
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
            raise RuntimeError("Variáveis ausentes: " + ", ".join(missing))

        config = OpenRouterConfig(
            api_key=api_key,
            base_url=base_url,
            http_referer=http_referer,
            x_title=x_title,
            embedding_model=model,
            batch_size=batch_size,
            max_retries=max_retries,
        )
        return cls(config)


    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        outputs = []
        for emb in self._embed_batches(texts):
            outputs.append(emb)
        return outputs

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


    def _embed_batches(self, texts: Sequence[str]) -> Iterable[List[float]]:
        for batch in _chunk_sequence(texts, self.config.batch_size):
            embeddings = self._request_embeddings(batch)
            for emb in embeddings:
                yield emb


    def _request_embeddings(self, batch: Sequence[str]) -> List[List[float]]:
        last_err = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=list(batch),
                )

                if not response or not getattr(response, "data", None):
                    raise RuntimeError("Resposta vazia ao pedir embeddings")

                return [item.embedding for item in response.data]

            except RateLimitError as e:
                last_err = e
                time.sleep(self.config.retry_base ** attempt)

            except APIError as e:
                last_err = e
                if getattr(e, "status_code", 0) in {500, 502, 503, 504}:
                    time.sleep(self.config.retry_base ** attempt)
                else:
                    raise RuntimeError(f"Erro API do OpenRouter: {e}") from e

            except Exception as e:
                raise RuntimeError(f"Falha inesperada: {e}") from e

        raise RuntimeError(f"Falha ao gerar embeddings após várias tentativas: {last_err}")


def _chunk_sequence(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    if size <= 0:
        raise ValueError("batch size precisa ser > 0")
    for start in range(0, len(seq), size):
        yield seq[start: start + size]
