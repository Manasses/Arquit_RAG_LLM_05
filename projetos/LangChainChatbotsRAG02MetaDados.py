# --- IMPORTAÇÕES ---
import os
# Loader para ler PDFs (Bulas)
from langchain_community.document_loaders import PyPDFLoader
# Divisor de texto em blocos menores (Chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings e LLM do Ollama (Processamento Local)
from langchain_ollama import OllamaEmbeddings, ChatOllama
# Banco vetorial para busca semântica
from langchain_community.vectorstores import Chroma
# Componentes para a Cadeia RAG moderna (LCEL)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURAÇÕES INICIAIS ---
# Definimos os nomes das bulas e a nova pasta para o banco de dados
caminhos_bulas = ["dipirona.pdf", "paracetamol.pdf"]
PASTA_CHROMA = "./chroma_db_bulas" 

# --- 2. CARREGAMENTO E ENRIQUECIMENTO DE METADADOS ---
documentos_totais = []

print("--- Iniciando carregamento das bulas ---")
for caminho in caminhos_bulas:
    if os.path.exists(caminho):
        loader = PyPDFLoader(caminho)
        paginas = loader.load()
        
        # Injetamos o nome do medicamento em cada página para o RAG saber de quem está falando
        nome_medicamento = caminho.replace(".pdf", "").capitalize()
        for p in paginas:
            p.metadata["medicamento"] = nome_medicamento
        
        documentos_totais.extend(paginas)
        print(f"✅ {nome_medicamento} carregado ({len(paginas)} páginas).")
    else:
        print(f"⚠️ Aviso: Arquivo {caminho} não encontrado.")

# --- 3. CHUNKING ESTRATÉGICO ---
# Dividimos o texto para que o modelo processe blocos médios com contexto compartilhado
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documentos_totais)
print(f"--- Total de chunks gerados: {len(chunks)} ---")

# --- 4. CLASSIFICAÇÃO SEMÂNTICA (TAGS DE CATEGORIA) ---
# Aqui adicionamos inteligência aos metadados para facilitar filtragens futuras
for chunk in chunks:
    texto = chunk.page_content.lower()
    if "contraindicação" in texto or "não devo usar" in texto:
        chunk.metadata["categoria"] = "contraindicacao"
    elif "posologia" in texto or "como devo usar" in texto:
        chunk.metadata["categoria"] = "posologia"
    elif "reações adversas" in texto or "males" in texto:
        chunk.metadata["categoria"] = "reacoes_adversas"
    else:
        chunk.metadata["categoria"] = "geral"

# --- 5. CRIAÇÃO DO BANCO VETORIAL (OLLAMA + CHROMA) ---
# Usamos o mxbai-embed-large para transformar texto em números (vetores)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

print("--- Criando/Atualizando Banco Vetorial Local ---")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PASTA_CHROMA
)

# --- 6. CONFIGURAÇÃO DO AGENTE (LLM + CADEIA) ---
# Usamos o Llama 3.2 3B como o "cérebro" que explica as bulas
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# Criamos o Template de Prompt focado em ser um "Farmacêutico Explicador"
template = """Você é um Agente Farmacêutico especializado.
Use o contexto abaixo para responder à pergunta de forma segura e clara.
Se a informação não estiver no contexto, responda que não encontrou na bula oficial.

Contexto: {context}
Pergunta: {question}

Resposta (seja atencioso e técnico):"""

prompt = ChatPromptTemplate.from_template(template)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Montagem da Cadeia RAG usando LCEL (Pipes)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 7. INTERAÇÃO E TESTE ---
pergunta = "Quais as reações adversas e contraindicações da dipirona?"

print(f"\n--- 💊 Pergunta: {pergunta} ---")
print("🔍 O Agente está analisando as bulas...\n")

# Usando stream para você ver a resposta sendo gerada em tempo real na sua CPU
print("🤖 RESPOSTA DO FARMACÊUTICO:")
for trecho in rag_chain.stream(pergunta):
    print(trecho, end="", flush=True)

# Recuperando as fontes para exibir os metadados enriquecidos
docs_origem = retriever.invoke(pergunta)
print("\n\n--- 📄 FONTES UTILIZADAS ---")
for d in docs_origem:
    med = d.metadata.get('medicamento', 'N/A')
    cat = d.metadata.get('categoria', 'N/A')
    pag = d.metadata.get('page', 'N/A')
    print(f"- Medicamento: {med} | Categoria: {cat} | Página: {pag}")

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
        resposta = rag_chain.invoke(user_input)               
        
        print(f"\n🤖 RESPOSTA:\n{resposta}")
        
        # Opcional: Mostrar as páginas de onde veio a informação
        docs = retriever.invoke(user_input)
        paginas = sorted(list(set([str(d.metadata.get('page', '?')) for d in docs])))
        print(f"\n📄 [Fontes: Páginas {', '.join(paginas)}]")
        print("-" * 40 + "\n")
        
    except Exception as e:
        print(f"❌ Ocorreu um erro durante a consulta: {e}")    