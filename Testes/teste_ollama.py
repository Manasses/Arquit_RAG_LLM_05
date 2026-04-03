from langchain_ollama import ChatOllama

# Configuração do modelo
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    base_url="http://localhost:11434" # Endereço padrão do Ollama
)

try:
    print("--- Testando conexão com Ollama ---")
    response = llm.invoke("Explique brevemente o que é a arquitetura Transformer.")
    print(f"🤖 Resposta do Agente:\n{response.content}")
except Exception as e:
    print(f"❌ Erro ao conectar: {e}. Certifique-se de que o Ollama está rodando!")