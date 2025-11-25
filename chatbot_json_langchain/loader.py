import json
from typing import Iterable, List
from langchain_core.documents import Document


def carregar_documentos_json(
    caminho_arquivo: str,
    *,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Carrega o JSON bruto e divide textos extensos em blocos menores.

    O OpenRouter limita o comprimento aceito por requisição de embedding. A divisão
    garante que conteúdos muito grandes (centenas de milhares de caracteres) não
    provoquem falhas silenciosas na API.
    """

    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        dados = json.load(f)

    documentos: List[Document] = []
    for item in dados:
        conteudo = item.get("conteudo", "") or ""
        metadata = {
            "url": item.get("url", ""),
            "titulo": item.get("titulo", ""),
        }

        if not conteudo.strip():
            continue

        documentos.extend(
            _dividir_em_chunks(conteudo, metadata, chunk_size, chunk_overlap)
        )

    return documentos


def _dividir_em_chunks(
    texto: str,
    metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterable[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser positivo")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap não pode ser negativo")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_size deve ser maior que chunk_overlap")

    documentos: List[Document] = []
    passo = chunk_size - chunk_overlap
    inicio = 0
    indice = 0

    while inicio < len(texto):
        fim = min(len(texto), inicio + chunk_size)
        fragmento = texto[inicio:fim]

        doc_metadata = dict(metadata)
        doc_metadata["chunk_index"] = indice
        doc_metadata["chunk_start"] = inicio
        doc_metadata["chunk_end"] = fim

        documentos.append(Document(page_content=fragmento, metadata=doc_metadata))

        indice += 1
        inicio += passo

    return documentos