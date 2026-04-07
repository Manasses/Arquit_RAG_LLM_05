# 🤖 Arquitetura RAG Avançada com LLMs Locais

Este repositório contém o desenvolvimento prático focado em arquiteturas **RAG (Retrieval-Augmented Generation)** para a construção de **Agentes de IA**. O foco principal é a implementação de pipelines de alta precisão utilizando modelos de linguagem executados 100% localmente.

---

## 💻 Especificações do Ambiente de Desenvolvimento
O projeto foi otimizado para um ambiente de alto desempenho em memória, visando privacidade e baixo custo operacional.
* **Sistema Operacional:** Windows 11
* **Hardware:** Intel Core i7-1255U (12ª Geração) | **40 GB RAM**
* **Runtime:** Python 3.13.5
* **IDE:** VS Code com ambiente virtual (`.venv`)
* **Engine de Inferência:** Ollama (Processamento focado em CPU/RAM)

---

## 📚 Tópicos e Competências Desenvolvidas

Neste repositório, exploramos o ciclo completo de implementação de um sistema de IA Generativa:

* **Estratégias Anti-Alucinação:** Integração de bases de conhecimento externas (PDFs) para garantir respostas baseadas em fatos.
* **Embeddings & Similaridade:** Uso de modelos especializados para transformação de texto em vetores e busca por similaridade de cosseno.
* **Gestão de Documentos:** Técnicas de *chunking* (divisão de texto), *overlapping* e enriquecimento semântico via metadados.
* **Pipelines de RAG:** Construção de fluxos declarativos utilizando **LangChain Expression Language (LCEL)**.
* **Vector Stores:** Indexação e persistência eficiente utilizando **ChromaDB**.
* **Re-rank Semântico:** Otimização da relevância das informações recuperadas para evitar a perda de contexto crítico.

---

## 📂 Projetos Implementados

### 1. RAG Foundation (Agente de Documentos)
Implementação da pipeline base de RAG. Focado no fluxo de extração de PDFs, criação de vetores e busca direta.
* **Destaque:** Lógica de persistência que verifica a existência do banco no SSD antes de reprocessar o documento, economizando recursos de CPU.

### 2. Agente Farmacêutico (Multidocumentos & Metadados)
Evolução para o tratamento de múltiplos arquivos simultâneos (Bulas de medicamentos).
* **Destaque:** Enriquecimento de metadados (`medicamento` e `categoria`) permitindo que o agente identifique exatamente de qual fonte a informação foi extraída, aumentando a confiabilidade técnica.

### 3. Agente de RH Corporativo (RAG + Re-ranking + Streamlit)
O projeto mais sofisticado, focado em políticas internas de RH (Férias, Home Office, Código de Conduta).
* **Inovação - Re-rank:** Implementação de um estágio duplo de recuperação. O sistema busca os 8 trechos mais próximos via vetores e utiliza a LLM para reordenar e selecionar os 3 mais relevantes.
* **Interface:** UI dinâmica desenvolvida em **Streamlit** com suporte a *streaming* de resposta em tempo real.

### 4. Notas Técnicas e Performance
* **Chunk Strategy:** Devido aos limites de contexto do modelo de embedding, utilizamos chunk_size=500 com chunk_overlap=100.
* **LCEL Architecture:** O uso de Pipes (|) no LangChain garante que o código seja modular e compatível com as versões mais recentes do Python.
* **Cache de Memória:** O uso de @st.cache_resource no Streamlit aproveita os 40GB de RAM do sistema para manter o banco vetorial carregado, reduzindo o tempo de resposta entre perguntas.

---

## 🦙 Configuração do Ollama

Para garantir a privacidade dos dados, todos os modelos rodam localmente via **Ollama**.

### 1. Instalação
* Download em: [ollama.com](https://ollama.com)

### 2. Modelos Utilizados
```bash
# Modelo principal para conversação e raciocínio (Llama 3.2 3B)
ollama pull llama3.2:3b

# Modelo especializado em Embeddings de alta qualidade
ollama pull mxbai-embed-large

### 3. Criação e Ativação da Venv
```bash
python -m venv venv
.\venv\Scripts\activate

### 4. Criação e Ativação da Venv
```bash
pip install langchain-ollama langchain-community langchain-core chromadb pypdf streamlit