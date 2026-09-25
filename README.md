# DCS RAG Agent

**Classe: LAB** (portfólio GenAI). **Não é AEGIS.** Não substitui o Hub `dcsproducer-hub`.  
Ver `GOVERNANCE.md` — opções: manter LAB ou privatizar.

Agente RAG com LangGraph — tools + memória (checkpointer) + multi-step + avaliação de trajetória.  
Sequência do starter `dcs-rag-starter`.

Fonte de verdade de projetos: planilha (`Lista_Projetos.xlsx`) + skill `pdf-wordpress-editor` + Hub canônico.

```
Pergunta → Agent Node → (Tools?) → Retrieval real (Chroma) / Cálculo → Resposta
         └─ MemorySaver (thread_id) ─ avaliação de trajetória
```

## O que esta versão entrega

| Item | Status |
|------|--------|
| Retrieval real via Chroma do starter | ✅ (fallback mock se DB ausente) |
| Checkpointer de memória (`MemorySaver` + `thread_id`) | ✅ |
| Avaliação de trajetória (tools usadas, correctness, passos) | ✅ |
| Tools: `search_knowledge_base`, `calculate` | ✅ |
| CLI + Streamlit | ✅ |

## Quick start

```bash
git clone https://github.com/producerdcs-cpu/dcs-rag-agent.git
cd dcs-rag-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 1. (Opcional) Ativar retrieval real
No `dcs-rag-starter`:
```bash
python -m src.rag_pipeline --ingest --data ./data
# copie a pasta chroma_db para este repo ou aponte CHROMA_PERSIST_DIR
```

### 2. CLI

```bash
python -m src.agent --ask "O que é RAG?"
python -m src.agent --ask "O que é RAG?" --thread sessao1 --trajectory
python -m src.agent --ask "O que é RAG?" --eval --expected-tools search_knowledge_base
```

### 3. Streamlit

```bash
streamlit run src/app_streamlit.py
```

Não commitar `.env` com chaves reais.

## Relação com a sequência

| Projeto | Papel |
|---------|-------|
| `dcs-rag-starter` | Pipeline RAG + avaliação de qualidade (faithfulness/relevance) |
| `dcs-rag-agent` (este) | Agente multi-step + tools + memória + trajetória |

Fonte de verdade de projetos e progresso: **planilha + skill pdf-wordpress-editor + Hub canônico**.

MIT © 2026 DcsProducer®
