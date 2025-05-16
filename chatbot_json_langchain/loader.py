import json
from langchain.schema import Document

def carregar_documentos_json(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    documentos = []
    for item in dados:
        conteudo = item.get("conteudo", "")
        metadata = {
            "titulo": item.get("titulo", ""),
            "id": item.get("id", "")
        }
        documentos.append(Document(page_content=conteudo, metadata=metadata))
    
    return documentos
