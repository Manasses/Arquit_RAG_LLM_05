import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. Configurações de Caminho e Modelo
CAMINHO_PDF = "GuiaEngenhariaPrompt.pdf"
PASTA_CHROMA = "./chroma_db_prompt"

# 2. Inicializa Embeddings (mxbai-embed-large)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Mostra onde o script está sendo executado
print(f"Diretório atual: {os.getcwd()}")

# --- LÓGICA DE PERSISTÊNCIA E CARREGAMENTO ---
if os.path.exists(PASTA_CHROMA) and os.listdir(PASTA_CHROMA):
    print(f"✅ Banco vetorial encontrado em '{PASTA_CHROMA}'. Carregando...")
    vectorstore = Chroma(
        persist_directory=PASTA_CHROMA, 
        embedding_function=embeddings
    )
else:
    print("--- Banco não encontrado ou vazio. Iniciando processamento do PDF ---")
    if not os.path.exists(CAMINHO_PDF):
        print(f"❌ ERRO: O arquivo '{CAMINHO_PDF}' não foi encontrado em {os.getcwd()}!")
        exit()
    
    # Carregamento e Divisão
    loader = PyPDFLoader(CAMINHO_PDF)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Criação do Banco e Persistência
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PASTA_CHROMA
    )
    print("✅ PDF processado e banco vetorial criado com sucesso!")

# 3. Configuração da LLM (Llama 3.2 3B)
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# 4. Definição da Cadeia RAG (LCEL)
template = """Você é um assistente especializado em Engenharia de Prompt. 
Use os seguintes pedaços de contexto recuperado para responder à pergunta. 
Se você não sabe a resposta, apenas diga que não sabe. Use no máximo três frases e mantenha a resposta concisa.

Contexto: {context}
Pergunta: {question}

Resposta:"""

prompt = ChatPromptTemplate.from_template(template)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. INTERAÇÃO TEXTUAL COM O USUÁRIO ---
print("\n" + "="*60)
print("🤖 AGENTE DE ENGENHARIA DE PROMPT (OLLAMA + RAG) ATIVO")
print("Digite sua pergunta ou 'sair' para encerrar o programa.")
print("="*60 + "\n")

while True:
    user_input = input("Sua pergunta: ")
    
    if user_input.lower() in ["sair", "exit", "quit", "parar"]:
        print("Encerrando o agente... Até logo!")
        break
        
    if not user_input.strip():
        continue

    try:
        print("🔍 Pesquisando no documento...")
        # Chamada da cadeia
        
        
        print(f"\n🤖 RESPOSTA:\n{resposta}")
        
        # Opcional: Mostrar as páginas de onde veio a informação
        docs = retriever.invoke(user_input)
        paginas = sorted(list(set([str(d.metadata.get('page', '?')) for d in docs])))
        print(f"\n📄 [Fontes: Páginas {', '.join(paginas)}]")
        print("-" * 40 + "\n")
        
    except Exception as e:
        print(f"❌ Ocorreu um erro durante a consulta: {e}")