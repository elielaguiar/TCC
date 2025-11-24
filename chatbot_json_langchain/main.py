import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from loader import carregar_documentos_json
from embedding_store import criar_chroma, carregar_chroma

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
CHUNK_SIZE = int(os.getenv("DOCS_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("DOCS_CHUNK_OVERLAP", "200"))

def criar_chatbot():
    persist_dir = "chroma_db"

    # Criar ou carregar Chroma
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print("🔄 Criando base vetorial...")
        if CHUNK_SIZE <= CHUNK_OVERLAP:
            raise ValueError("DOCS_CHUNK_SIZE deve ser maior que DOCS_CHUNK_OVERLAP.")

        documentos = carregar_documentos_json(
            "../dados/conteudo_paginas_tratado.json",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        print(f"✅ {len(documentos)} trechos preparados para indexação.")
        # print("DOCUMENTO EXEMPLO:", documentos[0])
        vectorstore = criar_chroma(documentos, persist_dir)
    else:
        print("✅ Carregando base vetorial existente...")
        vectorstore = carregar_chroma(persist_dir)

    retriever = vectorstore.as_retriever()
    # llm = ChatOpenAI(model="gpt-4o-mini")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Variável OPENROUTER_API_KEY não definida no ambiente.")

    referer = os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_SITE_URL")
    title = os.getenv("OPENROUTER_X_TITLE") or os.getenv("OPENROUTER_APP_TITLE")

    missing = []
    if not referer:
        missing.append("OPENROUTER_HTTP_REFERER ou OPENROUTER_SITE_URL")
    if not title:
        missing.append("OPENROUTER_X_TITLE ou OPENROUTER_APP_TITLE")

    if missing:
        raise RuntimeError(
            "Headers obrigatórios do OpenRouter não configurados: " + ", ".join(missing)
        )

    headers = {
        "HTTP-Referer": referer,
        "X-Title": title,
    }

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=headers or None,
    )

    # Prompt novo estilo LCEL
    prompt = ChatPromptTemplate.from_template("""
    Você é um assistente especializado. Responda com base no contexto abaixo.

    CONTEXTO:
    {context}

    PERGUNTA:
    {question}

    RESPOSTA:
    """)

    chain = (
    {
        "context": lambda x: "\n\n".join(
            doc.page_content for doc in retriever.invoke(x["question"])
        ),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
)

    return chain, retriever

if __name__ == "__main__":
    chain, retriever = criar_chatbot()

    while True:
        pergunta = input("Faça uma pergunta: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break

        resposta = chain.invoke({"question": pergunta})
        print("\n Bot:", resposta.content)

        # Buscar também os documentos usados
        docs = retriever.invoke(pergunta)
        fontes = [d.metadata.get("url", "Fonte desconhecida") for d in docs]
        print("Fontes:", fontes)
        print("-" * 50)