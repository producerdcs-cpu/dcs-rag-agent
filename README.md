# 🤖 DCS RAG Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DcsProducer®](https://img.shields.io/badge/DcsProducer®-GenAI-purple)](https://github.com/producerdcs-cpu)
[![Hub](https://img.shields.io/badge/Hub-Core%20AEGIS-blue)](https://core-hub-aegis.vercel.app/)

**Agente RAG com LangGraph** — tools + memória de conversa + raciocínio multi-step.

Sequência natural do [dcs-rag-starter](https://github.com/producerdcs-cpu/dcs-rag-starter).

```
Pergunta → Agent Node → (Tools?) → Retrieval / Cálculo → Resposta fundamentada
```

Parte do portfólio **GenAI DcsProducer®**.

**Portfólio central:** [Core Hub · AEGIS](https://core-hub-aegis.vercel.app/)

---

## ✨ O que este projeto demonstra

| Requisito de vaga                    | Como aparece aqui                          |
|--------------------------------------|--------------------------------------------|
| Agentes com LangGraph                | StateGraph + ToolNode + conditional edges  |
| Tools + function calling             | `search_knowledge_base`, `calculate`       |
| Memória de conversa                  | `add_messages` no state                    |
| Multi-step reasoning                 | Loop agent ↔ tools                         |
| Integração com RAG                   | Tool de retrieval (mock → real Chroma)     |
| CLI + Streamlit                      | Interface de demonstração                  |
| Configuração por ambiente            | `.env` com mock / ollama / groq / openai   |

---

## 🚀 Quick Start

```bash
git clone https://github.com/producerdcs-cpu/dcs-rag-agent.git
cd dcs-rag-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### CLI

```bash
python -m src.agent --ask "O que é RAG e como ele se relaciona com agentes?"
```

### Streamlit

```bash
streamlit run src/app_streamlit.py
```

### Docker

```bash
docker build -t dcs-rag-agent .
docker run -p 8501:8501 dcs-rag-agent
```

---

## 🛠️ Arquitetura do Agente

```
┌─────────────┐
│  Human msg  │
└──────┬──────┘
       ▼
┌─────────────┐     tool_calls?     ┌─────────────┐
│ Agent Node  │ ──────────────────► │  Tool Node  │
│ (LLM+tools) │ ◄────────────────── │ (execução)  │
└──────┬──────┘                     └─────────────┘
       │ no tools
       ▼
     [END]
```

- **State**: `messages` (histórico) + `context` opcional
- **Tools**: retrieval RAG + calculadora (expansível)
- **Conditional edge**: decide se precisa de ferramenta ou responde

---

## 📁 Estrutura

```
dcs-rag-agent/
├── src/
│   ├── agent.py            # StateGraph + tools + run_agent
│   └── app_streamlit.py    # UI de chat
├── .env.example
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🗺️ Roadmap

- [x] Skeleton LangGraph com tools e loop
- [ ] Conectar tool `search_knowledge_base` ao vector store real do `dcs-rag-starter`
- [ ] Memória persistente (checkpointer / Postgres)
- [ ] Avaliação de trajetória do agente (tool correctness + final answer)
- [ ] Hybrid search + re-ranking
- [ ] Observabilidade (LangSmith)
- [ ] Deploy (Railway / HF Spaces)

---

## 🔗 Relação com o starter

| Projeto              | Foco                              |
|----------------------|-----------------------------------|
| `dcs-rag-starter`    | Pipeline RAG clássico + avaliação |
| `dcs-rag-agent`      | Agente multi-step + tools         |

Reutilize o Chroma e o `eval_rag.py` do starter para manter consistência de métricas.

---

## ⚖️ Licença

MIT © 2026 **DcsProducer®** / [producerdcs-cpu](https://github.com/producerdcs-cpu)

---

**Anterior na sequência:** [dcs-rag-starter](https://github.com/producerdcs-cpu/dcs-rag-starter)  
**Hub:** [Core AEGIS](https://core-hub-aegis.vercel.app/)
