# rodar pip install streamlit

import os
import streamlit as st

# --- IMPORTAÇÕES DAS BIBLIOTECAS MODERNAS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# 1. CONFIGURAÇÕES GERAIS
# =========================
# Definimos as constantes para facilitar a manutenção do código
PERSIST_DIRECTORY = "./chroma_rh"
MODELO_LLM = "llama3.2:3b"
MODELO_EMBEDDINGS = "mxbai-embed-large"

# =========================
# 2. LEITURA E PROCESSAMENTO
# =========================

@st.cache_data # Cache do Streamlit para não ler os arquivos do SSD toda vez
def carregar_e_preparar_documentos():
    """
    Carrega PDFs, divide em chunks e enriquece com metadados.
    """
    caminhos = ["politica_ferias.pdf", "politica_home_office.pdf", "codigo_conduta.pdf"]
    documentos_base = []

    for caminho in caminhos:
        if os.path.exists(caminho):
            loader = PyPDFLoader(caminho)
            docs = loader.load()
            # Adicionamos metadados de origem
            for doc in docs:
                doc.metadata["documento"] = caminho
            documentos_base.extend(docs)

    # Chunking: Dividimos em blocos de 800 caracteres
    #splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documentos_base)

    # Enriquecimento de Metadados: Classificação automática
    for chunk in chunks:
        texto = chunk.page_content.lower()
        if "férias" in texto:
            chunk.metadata["categoria"] = "ferias"
        elif "home office" in texto or "remoto" in texto:
            chunk.metadata["categoria"] = "home_office"
        elif "conduta" in texto or "ética" in texto:
            chunk.metadata["categoria"] = "conduta"
        else:
            chunk.metadata["categoria"] = "geral"
    
    return chunks

# =========================
# 3. VECTOR STORE (OLLAMA)
# =========================

@st.cache_resource # Cache de recurso para manter o banco na RAM (40GB disponíveis!)
def obter_vectorstore(_chunks):
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS)
        
    # Se a pasta já existir, o Chroma apenas carrega. Se não, ele cria a partir dos chunks.
    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

# =========================
# 4. LÓGICA DE RERANKING
# =========================

def rerank_documentos(pergunta, documentos, llm):
    """
    Usa a inteligência da LLM para dar uma nota de 0 a 10 para cada trecho recuperado.
    Isso filtra falsos positivos da busca vetorial.
    """
    prompt_rerank = PromptTemplate.from_template(
        "Avalie de 0 a 10 a relevância do texto para a pergunta. Responda APENAS o número.\n"
        "Pergunta: {pergunta}\nTexto: {texto}"
    )
    
    # Criamos uma mini-cadeia para o rerank
    chain_rerank = prompt_rerank | llm | StrOutputParser()
    
    documentos_com_score = []

    for doc in documentos:
        # A LLM analisa cada trecho individualmente
        score_raw = chain_rerank.invoke({"pergunta": pergunta, "texto": doc.page_content})
        try:
            score = float(score_raw.strip())
        except:
            score = 0
        documentos_com_score.append((score, doc))

    # Ordenamos: o maior score (mais relevante) primeiro
    ordenados = sorted(documentos_com_score, key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ordenados]

# =========================
# 5. INTERFACE E PIPELINE
# =========================

st.set_page_config(page_title="Agente RH - Especialista IA", layout="wide")
st.title("🤖 Agente de RH — Consultoria Interna")

# Inicialização dos modelos
llm = ChatOllama(model=MODELO_LLM, temperature=0)
chunks = carregar_e_preparar_documentos()
vectorstore = obter_vectorstore(chunks)

pergunta = st.text_input("Como posso ajudar você hoje?")

if pergunta:
    with st.spinner("Buscando e reordenando políticas..."):
        # PASSO 1: Recuperação inicial (Busca 8 trechos)
        docs_iniciais = vectorstore.similarity_search(pergunta, k=8)

        # PASSO 2: Reranking (Refina os 8 para encontrar os melhores)
        docs_refinados = rerank_documentos(pergunta, docs_iniciais, llm)

        # PASSO 3: Geração da Resposta Final com os 3 melhores
        contexto_final = "\n\n".join([d.page_content for d in docs_refinados[:3]])
        
        prompt_final = ChatPromptTemplate.from_template("""
        Você é um assistente de RH. Responda estritamente com base no contexto abaixo.
        Contexto: {contexto}
        Pergunta: {pergunta}
        """)
        
        cadeia_final = prompt_final | llm | StrOutputParser()
        
        # Exibição com Streaming para melhor experiência do usuário
        st.subheader("Resposta do Agente")
        container_resposta = st.empty()
        resposta_completa = ""
        
        for chunk in cadeia_final.stream({"contexto": contexto_final, "pergunta": pergunta}):
            resposta_completa += chunk
            container_resposta.markdown(resposta_completa)

        # PASSO 4: Exibição das Fontes
        st.divider()
        st.subheader("Fontes de Dados (Após Reranking)")
        col1, col2, col3 = st.columns(3)
        for i, doc in enumerate(docs_refinados[:3]):
            with [col1, col2, col3][i]:
                st.info(f"**Fonte {i+1}**\n\nDoc: {doc.metadata.get('documento')}\n\nCat: {doc.metadata.get('categoria')}")
                st.caption(doc.page_content[:300] + "...")