import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from loader import carregar_documentos_json
from embedding_store import criar_chroma, carregar_chroma

load_dotenv()

def criar_chatbot():
    persist_dir = "chroma_db"

    # Criar ou carregar Chroma
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print("🔄 Criando base vetorial...")
        documentos = carregar_documentos_json("dados/conteudo_paginas_tratado.json")
        vectorstore = criar_chroma(documentos, persist_dir)
    else:
        print("✅ Carregando base vetorial existente...")
        vectorstore = carregar_chroma(persist_dir)

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini")

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