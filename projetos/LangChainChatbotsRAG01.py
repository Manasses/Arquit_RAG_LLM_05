import os
# Loader de documentos PDF
from langchain_community.document_loaders import PyPDFLoader
# Divisão de texto em blocos
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings e LLM do Ollama
from langchain_ollama import OllamaEmbeddings, ChatOllama
# Banco vetorial
from langchain_community.vectorstores import Chroma
# Cadeia RAG
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. Configurações de Caminho e Modelo
CAMINHO_PDF = "GuiaEngenhariaPrompt.pdf"
PASTA_CHROMA = "./chroma_db_prompt"

# 2. Carrega o PDF
print("--- Carregando PDF ---")

# Mostra onde o script está sendo executado
print(f"Diretório atual: {os.getcwd()}")

# Verifica se o arquivo existe antes de tentar carregar
if not os.path.exists(CAMINHO_PDF):
    print(f"❌ ERRO: O arquivo '{CAMINHO_PDF}' não foi encontrado no local acima!")
else:
    print(f"✅ Arquivo encontrado! Iniciando carregamento...")
    loader = PyPDFLoader(CAMINHO_PDF)

documents = loader.load()    

# 3. Divisão de texto (Chunking) - Essencial para o RAG funcionar
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# 4. Inicializa Embeddings (mxbai-embed-large)
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

# 5. Cria ou Carrega o banco vetorial
print("--- Criando/Carregando Banco Vetorial ---")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PASTA_CHROMA
)

# 6. Inicializa o modelo de linguagem local (Llama 3.2 3B)
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# 7. Definindo o Template do Prompt (O coração do especialista em Prompt)
template = """Você é um assistente especializado em Engenharia de Prompt. 
Use os seguintes pedaços de contexto recuperado para responder à pergunta. 
Se você não sabe a resposta, apenas diga que não sabe. Use no máximo três frases e mantenha a resposta concisa.

Contexto: {context}
Pergunta: {question}

Resposta:"""

prompt = ChatPromptTemplate.from_template(template)

# 8. Criando a Cadeia RAG moderna (Chain)
# Esta estrutura substitui o RetrievalQA
rag_chain = (
    {"context": vectorstore.as_retriever(search_kwargs={"k": 3}), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)