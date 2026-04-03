# Arquitetura RAG com LLMs Locais 🤖🚀

Este repositório contém o desenvolvimento prático do curso de construção de **Agentes de IA com Python**, focado em arquiteturas **RAG (Retrieval-Augmented Generation)** utilizando modelos de linguagem (LLMs) executados localmente.

---

## 📚 Tópicos do Curso

Neste treinamento, exploramos o ciclo completo de um Especialista em IA:

* **Fundamentos de LLM:** Compreensão de Transformers e sistemas generativos.
* **Estratégias Anti-Alucinação:** Integração de conhecimento externo para respostas precisas.
* **Personalização:** Comparação entre Fine-tuning e RAG.
* **Embeddings & Similaridade:** Transformação de documentos em vetores e busca por similaridade de cosseno.
* **Processamento de Documentos:** Técnicas de *chunking*, *overlapping* e gestão de metadados.
* **Pipelines de RAG:** Implementação de fluxos para documentos estruturados e não estruturados.
* **Vector Stores:** Uso de FAISS e ChromaDB para indexação eficiente.
* **Agentes LangChain:** Construção de interfaces conversacionais inteligentes.
* **Re-rank:** Otimização da relevância das informações recuperadas.

---

## 🦙 Configuração do Ollama (LLM Local)

Para este projeto, utilizamos o **Ollama** para garantir privacidade, baixo custo e execução 100% local.

### 1. Instalação
* Faça o download em: [ollama.com](https://ollama.com/download/windows)
* Após instalar, certifique-se de que o ícone do Ollama está ativo na bandeja do sistema.

### 2. Modelos Utilizados
Abra o seu terminal (PowerShell) e execute os seguintes comandos para baixar os modelos necessários:

* **Llama 3.2 (3B):** Modelo principal para conversação e raciocínio.
    ```bash
    ollama pull llama3.2:3b
    ```
* **mxbai-embed-large:** Modelo especializado em gerar Embeddings de alta qualidade.
    ```bash
    ollama pull mxbai-embed-large
    ```

---

## 🐍 Configuração do Ambiente Python (VS Code)

O projeto foi desenvolvido em **Windows 11** com foco em isolamento de dependências.
Especificações de Hardware (Referência):
CPU: Intel Core i7-1255U (12ª Geração)
RAM: 40 GB (Foco em processamento via CPU/RAM devido à GPU integrada).


### 1. Máquina Virtual (venv)
Para manter o ambiente limpo, criamos uma venv na raiz do projeto:
```powershell
python -m venv venv
.\venv\Scripts\activate

